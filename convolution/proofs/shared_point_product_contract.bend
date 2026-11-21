# Contract of the shared schedules with arranged public operands.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_matrix_products.bend as Matrix
import ../../lib/tensor_move.bend as Move
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/tensor_point_products.bend as Products
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_tiled_map.bend as TiledMap
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/kernel_gemm_output_groups.bend as Groups
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as Index
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_cell_room.bend as CellRoom
import ../../lib/proofs/tensor_count_completion.bend as Completion
import ../../lib/proofs/tensor_relabeled.bend as Relabel
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/tensor_grid_ranges.bend as Grids
import ../../lib/proofs/tensor_joined.bend as Joined
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_pieces_grouped_with.bend as Grouped
import ../../lib/proofs/tensor_pieces_model.bend as Model
import ../../lib/proofs/tensor_point_product_operands.bend as Operands
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_placement.bend as Placed
import ../../lib/proofs/tensor_point_product_result.bend as Result
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_point_product_tasks.bend as Shared
import ../../lib/proofs/tensor_point_product_values.bend as Values
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/tensor_shape_refinement.bend as Shape
import ../../lib/proofs/tensor_storage_count.bend as Count
import ../../lib/proofs/tensor_shared_product_placement.bend as Placement
import ../../lib/proofs/tensor_shared_product_samples.bend as Span
import ../../lib/proofs/tensor_tiled_padded_cells.bend as Padded
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/tensor_tiled_relane.bend as LaneLayout
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./product_values.bend as Logical
import ./shared_point_product_lanes.bend as Lanes
import ./shared_point_product_result.bend as Leaves

def Reads(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+bias: Bias,
  +rows: U32,+inner: U32,+channels: U32,resources: Point.ResourceContent) -> Data:
  Point.ResourceContent{+input,+offset} = resources
  Gemm.Both<Gemm.Both<{Buffer.content_length(~Tile.Tile8,input) == Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))) : Nat},
    Contract.Values(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},input)>,
    Gemm.Both<{Buffer.content_length(~Tile.Tile8,offset) == U32.to_nat(Gemm.blocks(channels)) : Nat},
      Contract.Values(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},offset)>>

def whole(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+bias: Bias,
  +rows: U32,+inner: U32,+channels: U32,+resource: Point.SharedResource,
  +reads: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(resource))) ->
  Lanes.WholeOperands<Left,left_value,Bias,bias_value,left,bias,rows,inner,channels>:
  Point.SharedResource{+input,+offset} = resource
  Gemm.Both{Gemm.Both{+input_len,+input_values},Gemm.Both{+offset_len,+offset_values}} = reads
  +input_size = Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)))
  +offset_size = U32.to_nat(Gemm.blocks(channels))
  +input_length = Buffer.length(~Tile.Tile8,input)
  +offset_length = Buffer.length(~Tile.Tile8,offset)
  +input_count = Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,input))
  +offset_count = Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,offset))
  Lanes.WholeOperands{input,offset,
    Equal.trans(Nat,input_length,input_count,input_size,Equal.sym(Nat,input_count,input_length,Buffer.content_length_of(~Tile.Tile8,input)),input_len),input_values,
    Equal.trans(Nat,offset_length,offset_count,offset_size,Equal.sym(Nat,offset_count,offset_length,Buffer.content_length_of(~Tile.Tile8,offset)),offset_len),offset_values}

type BoundsOf<-limits: Machine.Limits,-rows: U32,-inner: U32,-channels: U32> is Data:
  BoundsOf{inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    read_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weights_room: {Nat.is_le(Span.weight_cells(inner,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    bias_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(Nat.mul(1n,U32.to_nat(rows)),U32.to_nat(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    span: Values.Span<limits,inner>,whole: {U32.is_zero((inner % 16 : U32)) == True{} : Bool}}

def weight_size(+rows: U32,+inner: U32,+channels: U32,+first: U32,+count: U32,+limits: Machine.Limits,+clip: Shared.Clipped<limits,rows,channels,first,count>,
  +span: Span.Span<rows,inner,channels,first,count>,
  +room: {Nat.is_le(Span.weight_cells(inner,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(Shared.clipped_extent(rows,channels,first,count))),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Shared.Clipped{+extent,+extent_room,+positive,+exact} = clip
  Span.Span{+groups,+start_exact,+length_exact,+inside} = span
  +local = U32.to_nat(Gemm.blocks(Shared.clipped_extent(rows,channels,first,count)))
  +width = U32.to_nat(inner)
  +wanted = Nat.mul(local,width)
  +known = Span.weight_cells(inner,channels)
  +counted = Span.group_count(rows,count)
  +global = U32.to_nat(Gemm.blocks(channels))
  +low = Span.start_group(rows,first)
  +covered = Order.transitive(counted,Nat.add(low,counted),global,Order.right_le(low,counted),groups)
  Order.transitive(wanted,known,U32.to_nat(Machine.capacity(limits)),
    %Equal.sym(Nat,local,counted,exact) : {Nat.is_le(Nat.mul(_,width),known) == True{} : Bool}
    Index.mul_le(counted,global,width,covered),room)

type MaskLayout<-rows: U32,-channels: U32> is Data:
  MaskLayout{layout: Padded.Layout,geometry: {Padded.layout_geometry(layout) == Span.output_geometry(rows,channels) : Cells.Cells},
    count: {Padded.layout_total(layout) == Placement.output_cells(rows,channels) : Nat}}

def mask_layout(+rows: U32,+channels: U32,evidence: MaskLayout<rows,channels>) -> Padded.Layout:
  MaskLayout{layout,geometry,count} = evidence
  layout

def target_of(~Left: Data,~Weight: Data,~Bias: Data,+left: Left,+weight: Weight,+bias: Bias,+rows: U32,+inner: U32,+channels: U32) ->
  Representation.Lifted<Logical.Shared<Left,Weight,Bias>>:
  Representation.Lifted{Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},Span.output_geometry(rows,channels)}

def first_aligned(+rows: U32,+channels: U32,+first: U32,+count: U32,interval: Shared.GroupInterval<first,count,rows,channels>) ->
  {U32.to_nat(first) == Nat.mul(Span.start_group(rows,first),U32.to_nat(rows)) : Nat}:
  Shared.GroupInterval{lower,length,first_group,aligned,groups,cells,positive,inside} = interval
  aligned

def clip_group_room(+limits: Machine.Limits,+rows: U32,+channels: U32,+first: U32,+count: U32,
  clip: Shared.Clipped<limits,rows,channels,first,count>) ->
  {Nat.is_le(U32.to_nat(Gemm.blocks(Shared.clipped_extent(rows,channels,first,count))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Shared.Clipped{exact,+room,positive,groups} = clip
  Groups.groups_fit(limits,Shared.clipped_extent(rows,channels,first,count),room)

def seal_leaf(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+resource: Point.SharedResource,
  +reads: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(resource)),
  +bounds: BoundsOf<limits,rows,inner,channels>,+layout: Placement.Layout<limits,rows,channels,grid>,+mask: MaskLayout<rows,channels>,
  +fits: Runs.Fits<U32.to_nat(rows),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),Placement.output_cells(rows,channels)>,
  +window: Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Weight>,Representation.lifted(~Weight,~weight_value),
    Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},holding,Span.start_word(rows,inner,first),
      (Span.start_word(rows,inner,first) + Span.span_word(rows,inner,count) : U32),Span.weight_cells(inner,channels)>,
  +result: Shared.TaskModel(),
  +holds: {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
    Point.rectangle_model(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),Placement.leaf_stage(rows,inner,channels,grid,first,count,holding,resource)) :
    Model.Content<Tile.Tile8>}) ->
  Result.Checked<Representation.Lifted<Logical.Shared<Left,Weight,Bias>>,Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value)),
    target_of(~Left,~Weight,~Bias,left,weight,bias,rows,inner,channels),Padded.layout_padding(mask_layout(rows,channels,mask)),
    limits,Pieces.ranges_of_grid(grid),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),result>:
  Point.SharedResource{+input,+offset} = resource
  Gemm.Both{Gemm.Both{+input_len,+input_values},Gemm.Both{+offset_len,+offset_values}} = reads
  BoundsOf{+inner_room,+input_room,+weights_room,+bias_room,+output_room,+block,+complete} = bounds
  Placement.Layout{+grid_facts,+channel_room,+rows_positive,+rows_room,+cells,+room,+run} = layout
  MaskLayout{+padded,+geometry_exact,+area} = mask
  Runs.Fits{+low,+high,+low_exact,+high_exact,+positive,+inside} = fits
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.shared_geometry(work,first,count)
  +interval = Shared.group_interval(rows,channels,Machine.half_word(limits),rows_positive,Limits.double_bound(limits,U32.to_nat(rows),rows_room),Limits.half_word_fits(limits),first,count,fits)
  +clip = Shared.clipped_facts(limits,rows,channels,first,count,channel_room,interval)
  +span = Span.weighted_span(limits,rows,inner,channels,first,count,weights_room,interval)
  +length = Shared.clipped_cells(limits,rows,channels,first,count,interval,clip)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +first_cell = U32.to_nat(first)
  +rect_cells = Fragments.rectangle_cells(rectangle)
  +width = U32.to_nat(count)
  +end = Nat.add(first_cell,width)
  +end_room = Placement.layout_inside(limits,rows,channels,grid,layout,first,count,fits)
  +read_room = Placement.leaf_space(limits,rows,inner,channels,grid,first,count,rows_positive,rows_room,end_room,length,clip)
  +compute_room = {%length : {Nat.is_le(_,capacity) == True{} : Bool}
    Order.transitive(width,end,capacity,Order.right_le(first_cell,width),end_room) : {Nat.is_le(rect_cells,capacity) == True{} : Bool}}
  +extent = Shared.clipped_extent(rows,channels,first,count)
  +pair_room = {Operands.PairSpace{clip_group_room(limits,rows,channels,first,count,clip),inner_room,weight_size(rows,inner,channels,first,count,limits,clip,span,weights_room)} :
    Operands.PairSpace<limits,extent,inner>}
  +native_room = {Lanes.RunRoom{rows_room,compute_room,pair_room,block} : Lanes.RunRoom<limits,rows,inner,extent>}
  +covered = {%length : {Nat.is_le(Nat.add(first_cell,_),Padded.layout_total(padded)) == True{} : Bool}
    %Equal.sym(Nat,Padded.layout_total(padded),Placement.output_cells(rows,channels),area) : {Nat.is_le(end,_) == True{} : Bool} inside :
    {Nat.is_le(Nat.add(first_cell,rect_cells),Padded.layout_total(padded)) == True{} : Bool}}
  +leaf = {Leaves.Leaf{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,resource,reads),
    span,clip,window,input_room,weights_room,bias_room,output_room,TiledValues.grouped(limits,inner,inner_room),native_room,complete,
    first_aligned(rows,channels,first,count,interval),read_room,padded,geometry_exact,covered} : Leaves.Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>}
  +fragments = Shared.shared_leaf_laid(limits,rows,inner,channels,grid,grid_facts,channel_room,rows_positive,rows_room,cells,room,run,first,count,fits)
  Result.Checked{Placed.RectanglePlan{rectangle,Placement.leaf_stage(rows,inner,channels,grid,first,count,holding,resource),result,first,count,{==},length,{==},{==},read_room,holds,
      %length : Shared.Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(rectangle),first_cell,Nat.add(first_cell,_)) fragments},
    {==},Leaves.leaf_values(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)}

def copied_reads(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+bias: Bias,
  +rows: U32,+inner: U32,+channels: U32,+resource: Point.SharedResource,
  +owners: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(resource)),
  copies: Point.SharedCopies<resource>) ->
  Held.Both<Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,
      Point.resource_content(PartitionLaws.first(~Point.SharedResource,Shared.split_models(resource,copies)))),
    Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,
      Point.resource_content(PartitionLaws.second(~Point.SharedResource,Shared.split_models(resource,copies))))>:
  Point.SharedCopies{+low,+high,+exact,+left_content,+right_content} = copies
  Held.Both{
    %Equal.sym(Point.ResourceContent,Point.resource_content(low),Point.resource_content(resource),left_content) :
      Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,_) owners,
    %Equal.sym(Point.ResourceContent,Point.resource_content(high),Point.resource_content(resource),right_content) :
      Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,_) owners}

def CheckedShared(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +mask: MaskLayout<rows,channels>,+low: Nat,+high: Nat,+task: Shared.TaskModel()) -> Data:
  Result.Checked<Representation.Lifted<Logical.Shared<Left,Weight,Bias>>,Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value)),
    target_of(~Left,~Weight,~Bias,left,weight,bias,rows,inner,channels),Padded.layout_padding(mask_layout(rows,channels,mask)),limits,Pieces.ranges_of_grid(grid),low,high,task>

def first_ready(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+bias: Bias,
  +rows: U32,+inner: U32,+channels: U32,+low: Point.SharedResource,+high: Point.SharedResource,
  ready: Held.Both<Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(low)),
    Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(high))>) ->
  Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(low)):
  Held.Both{first,second} = ready
  first

def second_ready(~Left: Data,~left_value: Left -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,+left: Left,+bias: Bias,
  +rows: U32,+inner: U32,+channels: U32,+low: Point.SharedResource,+high: Point.SharedResource,
  ready: Held.Both<Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(low)),
    Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(high))>) ->
  Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(high)):
  Held.Both{first,second} = ready
  second
def checked_tree(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +bounds: BoundsOf<limits,rows,inner,channels>,+layout: Placement.Layout<limits,rows,channels,grid>,+mask: MaskLayout<rows,channels>,
  tasks: Pieces.Ranges,windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  +owners: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(resource)),
  runs: Runs.Run(tasks,U32.to_nat(rows),Placement.output_cells(rows,channels)),
  reads: Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),~Products.Shape,~Products.shared_reach,
    Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},Products.Shape{rows,inner,channels,grid},Span.weight_cells(inner,channels),tasks,windows),
  facts: Shared.SharedTree(Products.Shape{rows,inner,channels,grid},tasks,windows,resource)) ->
  CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,
    Runs.range_low(tasks),Runs.range_high(tasks),Shared.shared_results(Products.Shape{rows,inner,channels,grid},tasks,windows,resource,facts)):
  match tasks windows:
    case Pieces.Range{+first,+count} Windows.WindowLeaf{+holding,+window_first,+window_count}:
      Held.Both{+first_same,+rest} = reads
      Held.Both{+count_same,+window} = rest
      Grouped.TaskFact{+result,+returned,+exact,+regions} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      seal_leaf(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,first,count,holding,resource,
        owners,bounds,layout,mask,runs,window,result,
        %first_same : {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
          Point.rectangle_model(Point.shared_geometry(work,_,count),Placement.leaf_stage(rows,inner,channels,grid,_,count,holding,resource)) : Model.Content<Tile.Tile8>}
        %count_same : {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
          Point.rectangle_model(Point.shared_geometry(work,window_first,_),Placement.leaf_stage(rows,inner,channels,grid,window_first,_,holding,resource)) : Model.Content<Tile.Tile8>}
        Placement.leaf_contents(rows,inner,channels,grid,window_first,window_count,holding,resource,Grouped.TaskFact{result,returned,exact,regions}))
    case Pieces.Split{+lower,+upper} Windows.WindowFork{+left_windows,+right_windows}:
      Held.Both{+left_run,+rest} = runs
      Held.Both{+right_run,+glue} = rest
      Held.Both{+left_reads,+right_reads} = reads
      Held.Both{+left_facts,+right_facts} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      +copies = Point.shared_copies(resource)
      +split = Shared.split_models(resource,copies)
      +left_resource = PartitionLaws.first(~Point.SharedResource,split)
      +right_resource = PartitionLaws.second(~Point.SharedResource,split)
      +unit = U32.to_nat(rows)
      +bound = Placement.output_cells(rows,channels)
      +target = target_of(~Left,~Weight,~Bias,left,weight,bias,rows,inner,channels)
      +padding = Padded.layout_padding(mask_layout(rows,channels,mask))
      +left_model = Shared.shared_results(work,lower,left_windows,left_resource,left_facts)
      +right_model = Shared.shared_results(work,upper,right_windows,right_resource,right_facts)
      +first = Runs.range_low(lower)
      +middle = Runs.range_high(lower)
      +after = Runs.range_low(upper)
      +last = Runs.range_high(upper)
      +same = Equal.sym(Nat,middle,after,glue)
      +ready = copied_reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,resource,owners,copies)
      Result.checked_join(~Representation.Lifted<Logical.Shared<Left,Weight,Bias>>,~Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value)),
        target,padding,limits,Pieces.ranges_of_grid(grid),first,middle,last,left_model,right_model,
        Placement.run_order(lower,unit,bound,Runs.run_fits(lower,unit,bound,left_run)),
        %same : {Nat.is_le(_,last) == True{} : Bool} Placement.run_order(upper,unit,bound,Runs.run_fits(upper,unit,bound,right_run)),
        checked_tree(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,bounds,layout,mask,
          lower,left_windows,left_resource,first_ready(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,left_resource,right_resource,ready),left_run,left_reads,left_facts),
        %same : CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,_,last,right_model)
        checked_tree(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,bounds,layout,mask,
          upper,right_windows,right_resource,second_ready(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,left_resource,right_resource,ready),right_run,right_reads,right_facts))
    case Pieces.Range{+first,+count} Windows.WindowFork{+lower,+upper}:
      Empty.absurd(CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,
        U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),
        Shared.shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Range{first,count},Windows.WindowFork{lower,upper},resource,facts)),reads)
    case Pieces.Split{+lower,+upper} Windows.WindowLeaf{+holding,+first,+count}:
      Empty.absurd(CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,
        Runs.range_low(lower),Runs.range_high(upper),
        Shared.shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Split{lower,upper},Windows.WindowLeaf{holding,first,count},resource,facts)),reads)

def padded_shape(+limits: Machine.Limits,+rows: U32,+channels: U32,
  +room: {Nat.is_le(Shape.elements([1,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Padded.Layout:
  +rows_nat = U32.to_nat(rows)
  +size = U32.to_nat(channels)
  +bound = U32.to_nat(Machine.capacity(limits))
  +natural = {%Shape.elements_three(1,rows,channels) : {Nat.is_le(_,bound) == True{} : Bool} room :
    {Nat.is_le(Nat.mul(1n,Nat.mul(rows_nat,size)),bound) == True{} : Bool}}
  +counted = {%Index.multiply_commutative(rows_nat,size) : {Nat.is_le(_,bound) == True{} : Bool}
    %Addition.zero_right(Nat.mul(rows_nat,size)) : {Nat.is_le(_,bound) == True{} : Bool}
    natural : {Nat.is_le(Nat.mul(size,rows_nat),bound) == True{} : Bool}}
  Padded.Layout{limits,channels,rows,Grid.Outermost{},rows,1,Grid.Outermost{},Placement.output_cells(rows,channels),
    Public.Cover{counted,{==},{==},Public.Along{{==},{==}}}}

def shared_geometry(+rows: U32,+channels: U32) ->
  {Public.geometry_of([1,rows,channels],2n,Grid.Outermost{}) == Span.output_geometry(rows,channels) : Cells.Cells}:
  Equal.trans(Cells.Cells,Public.geometry_of([1,rows,channels],2n,Grid.Outermost{}),
    Cells.Cells{(1 * rows : U32),1,channels,Gemm.blocks(channels),Grid.Outermost{}},Span.output_geometry(rows,channels),
    Samples.matrix_geometry_of(1,rows,channels,Grid.Outermost{}),
    Equal.cong(U32,Cells.Cells,outer => Cells.Cells{outer,1,channels,Gemm.blocks(channels),Grid.Outermost{}},(1 * rows : U32),rows,Samples.one_times(rows)))

def input_geometry(+rows: U32,+inner: U32) ->
  {Public.geometry_of([1,rows,inner],2n,Grid.Innermost{}) == Samples.row_geometry(rows,inner) : Cells.Cells}:
  Equal.trans(Cells.Cells,Public.geometry_of([1,rows,inner],2n,Grid.Innermost{}),Samples.row_geometry((1 * rows : U32),inner),Samples.row_geometry(rows,inner),
    Samples.matrix_geometry_of(1,rows,inner,Grid.Innermost{}),
    Equal.cong(U32,Cells.Cells,outer => Samples.row_geometry(outer,inner),(1 * rows : U32),rows,Samples.one_times(rows)))

def seal_output(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +logical_room: {Nat.is_le(Shape.elements([1,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(Shape.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,Shape.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +task_model: Shared.TaskModel(),pieces: Pieces.Pieces(~Tile.Tile8),
  +placed: {pieces == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(Shared.shared_grid(rows,channels)),Products.zero(),Shared.pack_task(task_model)) : Pieces.Pieces(~Tile.Tile8)},
  +checked: CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,Shared.shared_grid(rows,channels),
    MaskLayout{padded_shape(limits,rows,channels,logical_room),{==},{==}},Runs.range_low(Pieces.ranges_of_grid(Shared.shared_grid(rows,channels))),
    Runs.range_high(Pieces.ranges_of_grid(Shared.shared_grid(rows,channels))),task_model)) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{pieces,[Gemm.blocks(channels),1,rows],Products.zero(),Tensors.Usable{},workers},2n,channels,Grid.Outermost{}},
      TiledMap.Padding{channels,Gemm.blocks(channels),rows,Grid.Outermost{}}))):
  Result.Checked{+plan,+same,+values} = checked
  +logical = {Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)} : Logical.Shared<Left,Weight,Bias>}
  +shape = {[Gemm.blocks(channels),1,rows] : List<&2,U32>}
  +dims = {[1,rows,channels] : List<&2,U32>}
  +padding_layout = padded_shape(limits,rows,channels,logical_room)
  +layout = Pieces.ranges_of_grid(Shared.shared_grid(rows,channels))
  +first = Runs.range_low(layout)
  +last = Runs.range_high(layout)
  +unit = U32.to_nat(rows)
  +divides = {Grids.Multiple{1n,Equal.sym(Nat,Nat.add(unit,0n),unit,Addition.zero_right(unit))} : Grids.Multiple<unit,unit>}
  +counted = Count.counted_within(limits,dims,logical_room,default_capacity)
  +size = Shared.shared_cells(rows,channels)
  +geometry = Public.geometry_of(dims,2n,Grid.Outermost{})
  Result.plan_contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    logical,dims,shape,limits,2n,channels,Grid.Outermost{},workers,padding_layout,
    %size : {Nat.is_lt(0n,_) == True{} : Bool} nonempty,{==},
    Equal.sym(Nat,Shape.elements(shape),Placement.output_cells(rows,channels),size),{==},
    Public.Laid{{==},{==},{==},counted,{==}},out_room,default_capacity,unit,
    Grids.grid_layout(limits,shape,Grid.piece_target(),out_room,nonempty,unit,divides),plan,
    %Equal.sym(Cells.Cells,geometry,Span.output_geometry(rows,channels),shared_geometry(rows,channels)) :
      Result.Masked(~Representation.Lifted<Logical.Shared<Left,Weight,Bias>>,~Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value)),
        Representation.Lifted{logical,_},Padded.layout_padding(padding_layout),limits,layout,first,last,plan)
    values,pieces,
    %Equal.sym(Shared.TaskModel(),Placed.planned_tasks(limits,layout,first,last,plan),task_model,same) :
      {pieces == Tensors.placed(~Tile.Tile8,layout,Products.zero(),Shared.pack_task(_)) : Pieces.Pieces(~Tile.Tile8)}
    placed)

def SharedResult(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+rows: U32,+inner: U32,+channels: U32,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Data:
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{pieces,[Gemm.blocks(channels),1,rows],Products.zero(),Tensors.Usable{},workers},
      2n,channels,Grid.Outermost{}},TiledMap.Padding{channels,Gemm.blocks(channels),rows,Grid.Outermost{}})))

def seal_scheduled(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical_room: {Nat.is_le(Shape.elements([1,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(Shape.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,Shape.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +bounds: BoundsOf<limits,rows,inner,channels>,+layout: Placement.Layout<limits,rows,channels,Shared.shared_grid(rows,channels)>,
  +resource: Point.SharedResource,+owners: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(resource)),
  +tasks: Pieces.Ranges,native: Pieces.Pieces(~Tile.Tile8),
  scheduled: Shared.ScheduledShared<Representation.Lifted<Weight>,Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},
    Shape.elements(Shared.weight_shape(inner,channels)),Products.Shape{rows,inner,channels,Shared.shared_grid(rows,channels)},tasks,resource,Shared.shared_grid(rows,channels),native>,
  run_of: Runs.RunOf(tasks,Pieces.ranges_of_grid(Shared.shared_grid(rows,channels)),U32.to_nat(rows),Placement.output_cells(rows,channels))) ->
  SharedResult(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,native):
  Shared.ScheduledShared{+windows,+reaches,+facts,+placed} = scheduled
  Held.Both{+run,Held.Both{+low_same,+high_same}} = run_of
  +grid = Shared.shared_grid(rows,channels)
  +ranges = Pieces.ranges_of_grid(grid)
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +source = {Representation.Lifted{weight,Samples.shared_geometry(inner,channels)} : Representation.Lifted<Weight>}
  +mask = {MaskLayout{padded_shape(limits,rows,channels,logical_room),{==},{==}} : MaskLayout<rows,channels>}
  +model = Shared.shared_results(work,tasks,windows,resource,facts)
  +contents = {%Shared.weight_cells(inner,channels) :
    Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),~Products.Shape,~Products.shared_reach,source,work,_,tasks,windows)
    reaches : Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),~Products.Shape,~Products.shared_reach,
      source,work,Span.weight_cells(inner,channels),tasks,windows)}
  seal_output(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,
    logical_room,out_room,out_nonempty,default_capacity,model,native,placed,
    %low_same : CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,_,Runs.range_high(ranges),model)
    %high_same : CheckedShared(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,mask,Runs.range_low(tasks),_,model)
    checked_tree(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,grid,bounds,layout,mask,
      tasks,windows,resource,owners,run,contents,facts))

def scheduled_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical_room: {Nat.is_le(Shape.elements([1,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(Shape.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,Shape.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_room: {Nat.is_le(Shape.elements(Shared.weight_shape(inner,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_nonempty: {Nat.is_lt(0n,Shape.elements(Shared.weight_shape(inner,channels))) == True{} : Bool},
  +bounds: BoundsOf<limits,rows,inner,channels>,+input: Buffer.Model<Tile.Tile8>,+offset: Buffer.Model<Tile.Tile8>,
  +owners: Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(Point.SharedResource{input,offset})),
  +weights_model: Model.Model<Tile.Tile8>,
  +covered: Held.Covered<Tile.Tile8,Representation.Lifted<Weight>,Representation.lifted(~Weight,~weight_value),
    Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},Move.held(~Tile.Tile8,Model.pack(~Tile.Tile8,weights_model)),0n,Shape.elements(Shared.weight_shape(inner,channels)),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(Shared.weight_shape(inner,channels))),Held.Ended{})>) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{
      Shared.shared_execution(Products.Shape{rows,inner,channels,Shared.shared_grid(rows,channels)},Tensors.grid_of(Shared.weight_shape(inner,channels)),
        Shared.shared_decision(rows,inner,channels,workers),Point.SharedResource{input,offset},Model.pack(~Tile.Tile8,weights_model)),
      [Gemm.blocks(channels),1,rows],Products.zero(),Tensors.Usable{},workers},2n,channels,Grid.Outermost{}},
      TiledMap.Padding{channels,Gemm.blocks(channels),rows,Grid.Outermost{}}))):
  +grid = Shared.shared_grid(rows,channels)
  +weight_grid = Tensors.grid_of(Shared.weight_shape(inner,channels))
  +decision = Shared.shared_decision(rows,inner,channels,workers)
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +resource = {Point.SharedResource{input,offset} : Point.SharedResource}
  +source = {Representation.Lifted{weight,Samples.shared_geometry(inner,channels)} : Representation.Lifted<Weight>}
  +total = Shape.elements(Shared.weight_shape(inner,channels))
  -weights = Model.pack(~Tile.Tile8,weights_model)
  +windowed = Shared.actual_shared_windows(~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),source,limits,rows,inner,channels,workers,
    rows_pos,rows_room,out_room,out_nonempty,weights_room,weights_nonempty,Model.pack(~Tile.Tile8,weights_model),covered)
  +tasks = Runs.decision_tasks(decision)
  +layout = Placement.prepared_layout(limits,rows,channels,channel_room,rows_pos,rows_room,out_room,out_nonempty)
  native = Shared.shared_execution(work,weight_grid,decision,resource,Model.pack(~Tile.Tile8,weights_model))
  seal_scheduled(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,
    default_capacity,logical_room,out_room,out_nonempty,bounds,layout,resource,owners,tasks,native,
    Shared.scheduled_decision(~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),source,total,work,weight_grid,decision,resource,weights,windowed),
    Shared.shared_run(limits,rows,inner,channels,workers,rows_room,out_room,out_nonempty))

def joined_buffer(+model: Model.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(shape),fill,Model.pack(~Tile.Tile8,model)))

def operand_contents(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+shape: List<&2,U32>,+model: Buffer.Model<Tile.Tile8>,
  reads: Held.Both<Contract.WellFormed<Tile.Tile8,shape,Buffer.content(~Tile.Tile8,model)>,Contract.Values(~Tile.Tile8,~Source,~value,source,Buffer.content(~Tile.Tile8,model))>) ->
  Gemm.Both<{Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)) == Shape.elements(shape) : Nat},
    Contract.Values(~Tile.Tile8,~Source,~value,source,Buffer.content(~Tile.Tile8,model))>:
  Held.Both{Contract.WellFormed{+extent,ready},+values} = reads
  Gemm.Both{extent,values}

def operand_read(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+model: Model.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +contract: Global.Contract(~Tile.Tile8,~Source,~value,source,Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})) ->
  Gemm.Both<{Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,joined_buffer(model,shape,fill))) == Shape.elements(shape) : Nat},
    Contract.Values(~Tile.Tile8,~Source,~value,source,Buffer.content(~Tile.Tile8,joined_buffer(model,shape,fill)))>:
  operand_contents(~Source,~value,source,shape,joined_buffer(model,shape,fill),
    Joined.joined_facts(~Tile.Tile8,~Source,~value,source,model,shape,fill,workers,contract,joined_buffer(model,shape,fill),
      Buffer.remodeled(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(shape),fill,Model.pack(~Tile.Tile8,model)))))

type SharedSizes<-limits: Machine.Limits,-rows: U32,-inner: U32,-channels: U32> is Data:
  SharedSizes{default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
    logical_room: {Nat.is_le(Shape.elements([1,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    row_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    inner_pos: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
    channel_pos: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
    row_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    whole: {U32.is_zero((inner % 16 : U32)) == True{} : Bool}}

type MatrixRooms<-limits: Machine.Limits,-rows: U32,-inner: U32,-channels: U32> is Data:
  MatrixRooms{input: {Nat.is_le(Nat.mul(U32.to_nat(inner),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def shared_sizes(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +row_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},+whole: {U32.is_zero((inner % 16 : U32)) == True{} : Bool},
  rooms: MatrixRooms<limits,rows,inner,channels>) -> SharedSizes<limits,rows,inner,channels>:
  MatrixRooms{+input,+output} = rooms
  +cap = U32.to_nat(Machine.capacity(limits))
  +rn = U32.to_nat(rows)
  +cn = U32.to_nat(channels)
  +n = U32.to_nat(inner)
  +total = Nat.mul(cn,rn)
  +swap = Index.multiply_commutative(cn,rn)
  +output_room = {%swap : {Nat.is_le(_,cap) == True{} : Bool} output : {Nat.is_le(Nat.mul(rn,cn),cap) == True{} : Bool}}
  +logical_room = {%Equal.sym(Nat,Shape.elements([1,rows,channels]),Nat.mul(1n,Nat.mul(rn,cn)),Shape.elements_three(1,rows,channels)) :
    {Nat.is_le(_,cap) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(rn,cn),0n),Nat.mul(rn,cn),Addition.zero_right(Nat.mul(rn,cn))) :
    {Nat.is_le(_,cap) == True{} : Bool} output_room : {Nat.is_le(Shape.elements([1,rows,channels]),cap) == True{} : Bool}}
  SharedSizes{default_capacity,logical_room,row_positive,inner_positive,channels_positive,
    Order.transitive(rn,Nat.mul(rn,cn),cap,Index.product_covers(rn,cn,channels_positive),output_room),
    Order.transitive(n,Nat.mul(n,rn),cap,Index.product_covers(n,rn,row_positive),input),
    Order.transitive(cn,total,cap,Index.product_covers(cn,rn,row_positive),output),whole}

def single_bias_size(+outputs: U32) -> {Shape.elements([Gemm.blocks(outputs)]) == U32.to_nat(Gemm.blocks(outputs)) : Nat}:
  Order.times_one(U32.to_nat(Gemm.blocks(outputs)))

def input_size(+rows: U32,+inner: U32) ->
  {Shape.elements([1,rows,1,Gemm.blocks(inner)]) == Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))) : Nat}:
  +groups = U32.to_nat(Gemm.blocks(inner))
  +length = Nat.mul(U32.to_nat(rows),groups)
  %Equal.sym(Nat,Nat.mul(groups,1n),groups,Order.times_one(groups)) :
    {Nat.add(Nat.mul(U32.to_nat(rows),Nat.add(_,0n)),0n) == length : Nat}
  %Equal.sym(Nat,Nat.add(groups,0n),groups,Addition.zero_right(groups)) : {Nat.add(Nat.mul(U32.to_nat(rows),_),0n) == length : Nat}
  Addition.zero_right(length)

def output_size(+rows: U32,+channels: U32,+positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +groups: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool}) ->
  {Nat.is_lt(0n,Shape.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool}:
  %Equal.sym(Nat,Shape.elements([Gemm.blocks(channels),1,rows]),Placement.output_cells(rows,channels),Shared.shared_cells(rows,channels)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows),groups,positive)

def joined_input_exact(+limits: Machine.Limits,+rows: U32,+inner: U32,+model: Model.Model<Tile.Tile8>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +room: {Nat.is_le(Shape.elements([1,rows,1,Gemm.blocks(inner)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Tensors.joined(~Tile.Tile8,(rows * Gemm.blocks(inner) : U32),Products.zero(),Model.pack(~Tile.Tile8,model)) ==
    Buffer.pack(~Tile.Tile8,joined_buffer(model,[1,rows,1,Gemm.blocks(inner)],Products.zero())) : Products.Cells()}:
  +shape = {[1,rows,1,Gemm.blocks(inner)] : List<&2,U32>}
  +cap = Machine.capacity(limits)
  +counted = Count.counted_within(limits,shape,room,default_capacity)
  +count = Global.count(shape,counted)
  +product = (rows * Gemm.blocks(inner) : U32)
  +area = Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)))
  +same = input_size(rows,inner)
  +exact = Arithmetic.product_exact(cap,Limits.capacity_fits(limits),rows,Gemm.blocks(inner),U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)),{==},{==},
    %same : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room)
  +count_exact = Global.whole_counted(shape,count,Global.accepted(shape,counted))
  +word = Equal.trans(U32,product,U32.from_nat(area),count,
    Equal.trans(U32,product,U32.from_nat(U32.to_nat(product)),U32.from_nat(area),Equal.sym(U32,U32.from_nat(U32.to_nat(product)),product,Arithmetic.native_roundtrip(product)),
      Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(product),area,exact)),
    Equal.trans(U32,U32.from_nat(area),U32.from_nat(U32.to_nat(count)),count,
      Equal.cong(Nat,U32,U32.from_nat,area,U32.to_nat(count),Equal.trans(Nat,area,Shape.elements(shape),U32.to_nat(count),
        Equal.sym(Nat,Shape.elements(shape),area,same),count_exact)),Arithmetic.native_roundtrip(count)))
  %Equal.sym(U32,product,count,word) : {Tensors.joined(~Tile.Tile8,_,Products.zero(),Model.pack(~Tile.Tile8,model)) == Buffer.pack(~Tile.Tile8,joined_buffer(model,shape,Products.zero())) : Products.Cells()}
  %Global.whole_width(shape,count,Global.accepted(shape,counted)) :
    {Tensors.joined(~Tile.Tile8,_,Products.zero(),Model.pack(~Tile.Tile8,model)) == Buffer.pack(~Tile.Tile8,joined_buffer(model,shape,Products.zero())) : Products.Cells()}
  Equal.sym(Products.Cells(),Buffer.pack(~Tile.Tile8,joined_buffer(model,shape,Products.zero())),Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(shape),Products.zero(),Model.pack(~Tile.Tile8,model)),Buffer.remodeled(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(shape),Products.zero(),Model.pack(~Tile.Tile8,model))))

def modeled_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  sizes: SharedSizes<limits,rows,inner,channels>,
  +lm: Model.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,
  +left_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[1,rows,1,Gemm.blocks(inner)],lf,Tensors.Usable{},lw}),
  +wm: Model.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +weight_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[Gemm.blocks(channels),inner],wf,Tensors.Usable{},ww}),
  +bm: Model.Model<Tile.Tile8>,+bf: Tile.Tile8,+bw: U32,
  +bias_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(channels)],bf,Tensors.Usable{},bw})) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    Matrix.scheduled_run(Matrix.SharedGroups{},1,rows,inner,channels,workers,
      Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[1,rows,1,Gemm.blocks(inner)],lf,Tensors.Usable{},lw},
      Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[Gemm.blocks(channels),inner],wf,Tensors.Usable{},ww},
      Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(channels)],bf,Tensors.Usable{},bw})):
  SharedSizes{+defaults,+logical_room,+row_positive,+inner_positive,+channel_positive,+rows_room,+inner_room,+channel_room,+whole} = sizes
  +left_shape = {[1,rows,1,Gemm.blocks(inner)] : List<&2,U32>}
  +weight_shape = Shared.weight_shape(inner,channels)
  +bias_shape = {[Gemm.blocks(channels)] : List<&2,U32>}
  +output_shape = {[Gemm.blocks(channels),1,rows] : List<&2,U32>}
  +logical_shape = {[1,rows,channels] : List<&2,U32>}
  +cap = Machine.capacity(limits)
  +max_cells = U32.to_nat(cap)
  +left_reader = {Representation.Lifted{left,Samples.row_geometry(rows,inner)} : Representation.Lifted<Left>}
  +right_reader = {Representation.Lifted{weight,Samples.shared_geometry(inner,channels)} : Representation.Lifted<Weight>}
  +bias_reader = {Representation.Lifted{bias,Samples.row_geometry(1,channels)} : Representation.Lifted<Bias>}
  +left_counted = Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    left_reader,Model.pack(~Tile.Tile8,lm),left_shape,lf,lw,left_contract)
  +weight_counted = Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),
    right_reader,Model.pack(~Tile.Tile8,wm),weight_shape,wf,ww,weight_contract)
  +input_room = Count.counted_room(limits,left_shape,left_counted,defaults)
  +weight_room = Count.counted_room(limits,weight_shape,weight_counted,defaults)
  +groups_room = Groups.groups_fit(limits,channels,channel_room)
  +groups_positive = Groups.groups_positive(limits,channels,channel_room,channel_positive)
  +weights_positive = {%Equal.sym(Nat,Shape.elements(weight_shape),Span.weight_cells(inner,channels),Shared.weight_cells(inner,channels)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner),groups_positive,inner_positive) : {Nat.is_lt(0n,Shape.elements(weight_shape)) == True{} : Bool}}
  +logical_positive = {%Equal.sym(Nat,Shape.elements(logical_shape),Nat.mul(1n,Nat.mul(U32.to_nat(rows),U32.to_nat(channels))),Shape.elements_three(1,rows,channels)) :
    {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(1n,Nat.mul(U32.to_nat(rows),U32.to_nat(channels)),{==},Order.product_positive(U32.to_nat(rows),U32.to_nat(channels),row_positive,channel_positive)) :
    {Nat.is_lt(0n,Shape.elements(logical_shape)) == True{} : Bool}}
  +out_room = CellRoom.cells_room(limits,logical_shape,2n,Grid.Outermost{},logical_positive,logical_room)
  +out_positive = output_size(rows,channels,row_positive,groups_positive)
  +bounds = {BoundsOf{inner_room,
    %input_size(rows,inner) : {Nat.is_le(_,max_cells) == True{} : Bool} input_room,
    %Shared.weight_cells(inner,channels) : {Nat.is_le(_,max_cells) == True{} : Bool} weight_room,
    groups_room,
    %Equal.trans(Nat,Shape.elements(logical_shape),Nat.mul(1n,Nat.mul(U32.to_nat(rows),U32.to_nat(channels))),Nat.mul(Nat.mul(1n,U32.to_nat(rows)),U32.to_nat(channels)),
        Shape.elements_three(1,rows,channels),Equal.sym(Nat,Nat.mul(Nat.mul(1n,U32.to_nat(rows)),U32.to_nat(channels)),Nat.mul(1n,Nat.mul(U32.to_nat(rows),U32.to_nat(channels))),
          Index.assoc(1n,U32.to_nat(rows),U32.to_nat(channels)))) : {Nat.is_le(_,max_cells) == True{} : Bool} logical_room,
    Values.block_span(limits,inner,inner_room,inner_positive,
      Bounds.as_nat_le(Gemm.block_tiles(),cap,%Equal.sym(U32,cap,Machine.capacity(Default.limits()),defaults) : {U32.is_le(Gemm.block_tiles(),_) == True{} : Bool} {==})),
    whole} : BoundsOf<limits,rows,inner,channels>}
  +input = joined_buffer(lm,left_shape,Products.zero())
  +offset = joined_buffer(bm,bias_shape,bf)
  +owners = {Gemm.Both{
    %input_size(rows,inner) : Gemm.Both<{Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,input)) == _ : Nat},
      Contract.Values(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),left_reader,Buffer.content(~Tile.Tile8,input))>
    operand_read(~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),left_reader,lm,left_shape,Products.zero(),lw,left_contract),
    %single_bias_size(channels) : Gemm.Both<{Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,offset)) == _ : Nat},
      Contract.Values(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),bias_reader,Buffer.content(~Tile.Tile8,offset))>
    operand_read(~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),bias_reader,bm,bias_shape,bf,bw,bias_contract)} :
    Reads(~Left,~left_value,~Bias,~bias_value,left,bias,rows,inner,channels,Point.resource_content(Point.SharedResource{input,offset}))}
  -weights = Model.pack(~Tile.Tile8,wm)
  -inputs = Model.pack(~Tile.Tile8,lm)
  -native_bias = Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(bias_shape),bf,Model.pack(~Tile.Tile8,bm))
  +grid = Shared.shared_grid(rows,channels)
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +decision = Shared.shared_decision(rows,inner,channels,workers)
  +weight_grid = Tensors.grid_of(weight_shape)
  %Buffer.remodeled(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(bias_shape),bf,Model.pack(~Tile.Tile8,bm))) :
    SharedResult(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,Products.shared(rows,inner,channels,workers,inputs,weights,_))
  %Equal.sym(Products.Cells(),Tensors.joined(~Tile.Tile8,(rows * Gemm.blocks(inner) : U32),Products.zero(),inputs),Buffer.pack(~Tile.Tile8,input),
    joined_input_exact(limits,rows,inner,lm,defaults,input_room)) :
    SharedResult(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,
      Products.shared_tasks(work,weight_grid,decision,_,weights,Buffer.pack(~Tile.Tile8,offset)))
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),Products.shared_tasks(work,weight_grid,decision,Buffer.pack(~Tile.Tile8,input),weights,Buffer.pack(~Tile.Tile8,offset)),
    Shared.shared_execution(work,weight_grid,decision,Point.SharedResource{input,offset},weights),Shared.executed(rows,inner,channels,grid,weight_grid,decision,input,offset,weights)) :
    SharedResult(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,_)
  scheduled_contract(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,defaults,
    logical_room,out_room,out_positive,channel_room,row_positive,rows_room,weight_room,weights_positive,bounds,input,offset,owners,wm,
    Windows.shaped_covered(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),right_reader,weight_shape,weights,wf,Tensors.Usable{},ww,
      Gemm.Both{Global.Usable{wm,wf,ww,{==}},weight_contract}))

def shaped_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +sizes: SharedSizes<limits,rows,inner,channels>,-input: Tensors.Tensor<Tile.Tile8>,-weights: Tensors.Tensor<Tile.Tile8>,-offset: Tensors.Tensor<Tile.Tile8>,
  input_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},
    [1,rows,1,Gemm.blocks(inner)],input),
  weights_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},
    [Gemm.blocks(channels),inner],weights),
  offset_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},
    [Gemm.blocks(channels)],offset)) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],Matrix.scheduled_run(Matrix.SharedGroups{},1,rows,inner,channels,workers,input,weights,offset)):
  Gemm.Both{Global.Usable{+lm,+lf,+lw,+input_eq},+read_input} = input_proof
  Gemm.Both{Global.Usable{+wm,+wf,+ww,+weight_eq},+read_weights} = weights_proof
  Gemm.Both{Global.Usable{+bm,+bf,+bw,+offset_eq},+read_offset} = offset_proof
  -input_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[1,rows,1,Gemm.blocks(inner)],lf,Tensors.Usable{},lw} : Tensors.Tensor<Tile.Tile8>}
  -weight_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[Gemm.blocks(channels),inner],wf,Tensors.Usable{},ww} : Tensors.Tensor<Tile.Tile8>}
  -bias_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(channels)],bf,Tensors.Usable{},bw} : Tensors.Tensor<Tile.Tile8>}
  +logical = {Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)} : Logical.Shared<Left,Weight,Bias>}
  +shape = {[1,rows,channels] : List<&2,U32>}
  %input_eq : Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),logical,shape,
    Matrix.scheduled_run(Matrix.SharedGroups{},1,rows,inner,channels,workers,_,weights,offset))
  %weight_eq : Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),logical,shape,
    Matrix.scheduled_run(Matrix.SharedGroups{},1,rows,inner,channels,workers,input_cells,_,offset))
  %offset_eq : Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),logical,shape,
    Matrix.scheduled_run(Matrix.SharedGroups{},1,rows,inner,channels,workers,input_cells,weight_cells,_))
  modeled_contract(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,sizes,lm,lf,lw,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,input_cells,input,input_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},_) read_input,
    wm,wf,ww,%Equal.sym(Tensors.Tensor<Tile.Tile8>,weight_cells,weights,weight_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},_) read_weights,
    bm,bf,bw,%Equal.sym(Tensors.Tensor<Tile.Tile8>,bias_cells,offset,offset_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},_) read_offset)

def input_relabel(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+limits: Machine.Limits,+rows: U32,+inner: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rows_pos: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+groups_pos: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(inner))) == True{} : Bool},
  -input: Tensors.Tensor<Tile.Tile8>,+shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry(rows,inner)},[1,rows,Gemm.blocks(inner)],input)) ->
  Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},
    [1,rows,1,Gemm.blocks(inner)],Tensors.relabeled(~Tile.Tile8,input,[1,rows,1,Gemm.blocks(inner)])):
  +source = {[1,rows,Gemm.blocks(inner)] : List<&2,U32>}
  +target = {[1,rows,1,Gemm.blocks(inner)] : List<&2,U32>}
  +reader = {Representation.Lifted{left,Samples.row_geometry(rows,inner)} : Representation.Lifted<Left>}
  +counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),reader,source,input,shaped)
  +count = Global.count(source,counted)
  +accepted = Global.accepted(source,counted)
  +size = Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)))
  +source_size = Equal.trans(Nat,Shape.elements(source),Nat.mul(1n,size),size,Shape.elements_three(1,rows,Gemm.blocks(inner)),Addition.zero_right(size))
  +same = Equal.trans(Nat,Shape.elements(source),size,Shape.elements(target),source_size,Equal.sym(Nat,Shape.elements(target),size,input_size(rows,inner)))
  +extent = Equal.trans(Nat,size,Shape.elements(source),U32.to_nat(count),Equal.sym(Nat,Shape.elements(source),size,source_size),Global.whole_counted(source,count,accepted))
  +positive = {%extent : {Nat.is_lt(0n,_) == True{} : Bool} Order.product_positive(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner)),rows_pos,groups_pos) :
    {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}}
  Relabel.relabeled(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),reader,limits,source,target,input,shaped,
    count,accepted,Relabel.counted_same(source,target,counted,same),Arithmetic.word_positive_nonzero(count,positive),default_capacity)

def arranged_contract(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  sizes: SharedSizes<limits,rows,inner,channels>,
  input: Tiled.Tiled,input_proof: LaneLayout.Arranged(~Left,~left_value,left,[1,rows,inner],input,2n,Grid.Innermost{}),
  weights: Tiled.Tiled,weights_proof: LaneLayout.Arranged(~Weight,~weight_value,weight,[channels,inner],weights,0n,Grid.Outermost{}),
  offset: Tiled.Tiled,offset_proof: LaneLayout.Arranged(~Bias,~bias_value,bias,[channels],offset,0n,Grid.Innermost{})) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    Matrix.scheduled_lines(Matrix.SharedGroups{},1,rows,inner,channels,workers,Matrix.innermost(input,2n),
      weights,Matrix.innermost(offset,0n))):
  SharedSizes{+default_capacity,+logical_room,+rows_positive,+inner_positive,+outputs_positive,+rows_room,+inner_room,+channels_room,+whole} = sizes
  Gemm.Both{input_form,+input_tiles} = input_proof
  Gemm.Both{weight_form,+weight_tiles} = weights_proof
  Gemm.Both{offset_form,+offset_tiles} = offset_proof
  +left_shape = {[1,rows,inner] : List<&2,U32>}
  +right_shape = {[channels,inner] : List<&2,U32>}
  +bias_shape = {[channels] : List<&2,U32>}
  -input_cells = Matrix.innermost(input,2n)
  -weight_cells = Matrix.weight_cells(weights)
  -bias_cells = Matrix.innermost(offset,0n)
  +input_ready = {%input_geometry(rows,inner) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,_},[1,rows,Gemm.blocks(inner)],input_cells)
    Public.cells_shaped(~Left,~left_value,left,left_shape,2n,Grid.lanes_at(left_shape,2n),Grid.Innermost{},Matrix.innermost(input,2n),input_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry(rows,inner)},[1,rows,Gemm.blocks(inner)],input_cells)}
  +weight_ready = {%Samples.shared_geometry_of(inner,channels) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,_},[Gemm.blocks(channels),inner],weight_cells)
    Public.cells_shaped(~Weight,~weight_value,weight,right_shape,0n,Grid.lanes_at(right_shape,0n),Grid.Outermost{},Matrix.weight_cells(weights),weight_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Weight>,~Representation.lifted(~Weight,~weight_value),Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},
      [Gemm.blocks(channels),inner],weight_cells)}
  +bias_ready = {%Samples.bias_geometry_of(channels) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,_},[Gemm.blocks(channels)],bias_cells)
    Public.cells_shaped(~Bias,~bias_value,bias,bias_shape,0n,Grid.lanes_at(bias_shape,0n),Grid.Innermost{},Matrix.innermost(offset,0n),offset_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Bias>,~Representation.lifted(~Bias,~bias_value),Representation.Lifted{bias,Samples.row_geometry(1,channels)},
      [Gemm.blocks(channels)],bias_cells)}
  shaped_contract(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,
    SharedSizes{default_capacity,logical_room,rows_positive,inner_positive,outputs_positive,rows_room,inner_room,channels_room,whole},
    Tensors.relabeled(~Tile.Tile8,input_cells,[1,rows,1,Gemm.blocks(inner)]),weight_cells,bias_cells,
    input_relabel(~Left,~left_value,left,limits,rows,inner,default_capacity,rows_positive,Groups.groups_positive(limits,inner,inner_room,inner_positive),input_cells,input_ready),
    weight_ready,bias_ready)

def Lines(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +left: Left,+weight: Weight,+bias: Bias,+rows: U32,+inner: U32,+channels: U32,+workers: U32,+route: Matrix.Schedule,
  input: Tiled.Tiled,weights: Tiled.Tiled,offset: Tiled.Tiled) -> Data:
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],
    Matrix.scheduled_lines(route,1,rows,inner,channels,workers,Matrix.innermost(input,2n),
      weights,Matrix.innermost(offset,0n)))

def arranged_tiles(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+shape: List<&2,U32>,-input: Tiled.Tiled,+axis: Nat,+position: Grid.Position,
  arranged: LaneLayout.Arranged(~Source,~value,source,shape,input,axis,position)) ->
  Public.Contract(~Source,~value,source,shape,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(input,axis,position)),axis,Grid.lanes_at(shape,axis),position}):
  Gemm.Both{form,values} = arranged
  values

def choose_lines(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  selected: Bool,+left: Left,+weight: Weight,+bias: Bias,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +row_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +channel_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},+rooms: MatrixRooms<limits,rows,inner,channels>,
  +decision: {(U32.is_zero((inner % 16 : U32)) && U32.is_le(Gemm.blocks(rows),Gemm.blocks(channels))) == selected : Bool},
  input: Tiled.Tiled,+input_proof: LaneLayout.Arranged(~Left,~left_value,left,[1,rows,inner],input,2n,Grid.Innermost{}),
  weights: Tiled.Tiled,+weights_proof: LaneLayout.Arranged(~Weight,~weight_value,weight,[channels,inner],weights,0n,Grid.Outermost{}),
  offset: Tiled.Tiled,+offset_proof: LaneLayout.Arranged(~Bias,~bias_value,bias,[channels],offset,0n,Grid.Innermost{}),
  window: @input: Tiled.Tiled -> @+input_proof: LaneLayout.Arranged(~Left,~left_value,left,[1,rows,inner],input,2n,Grid.Innermost{}) ->
    @weights: Tiled.Tiled -> @+weights_proof: LaneLayout.Arranged(~Weight,~weight_value,weight,[channels,inner],weights,0n,Grid.Outermost{}) ->
    @offset: Tiled.Tiled -> @+offset_proof: LaneLayout.Arranged(~Bias,~bias_value,bias,[channels],offset,0n,Grid.Innermost{}) ->
    Lines(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,Matrix.Window{},input,weights,offset)) ->
  Public.Contract(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)},[1,rows,channels],Matrix.lines(1,rows,inner,channels,workers,0n,input,weights,offset)):
  match selected:
    case False{}:
      %Equal.sym(Bool,(U32.is_zero((inner % 16 : U32)) && U32.is_le(Gemm.blocks(rows),Gemm.blocks(channels))),False{},decision) :
        Lines(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,
          Bool.pick(Matrix.Schedule,_,Matrix.SharedGroups{},Matrix.Window{}),input,weights,offset)
      window(input,input_proof,weights,weights_proof,offset,offset_proof)
    case True{}:
      %Equal.sym(Bool,(U32.is_zero((inner % 16 : U32)) && U32.is_le(Gemm.blocks(rows),Gemm.blocks(channels))),True{},decision) :
        Lines(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,rows,inner,channels,workers,
          Bool.pick(Matrix.Schedule,_,Matrix.SharedGroups{},Matrix.Window{}),input,weights,offset)
      arranged_contract(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,left,weight,bias,limits,rows,inner,channels,workers,
        shared_sizes(limits,rows,inner,channels,default_capacity,row_positive,inner_positive,channel_positive,
          Boolean.left(U32.is_zero((inner % 16 : U32)),U32.is_le(Gemm.blocks(rows),Gemm.blocks(channels)),decision),rooms),
        input,input_proof,weights,weights_proof,offset,offset_proof)
