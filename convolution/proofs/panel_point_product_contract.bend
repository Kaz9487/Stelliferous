# Contract of the image panels with arranged public operands.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/tensor_point_products.bend as Products
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/row_major_index.bend as Index
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_full_values.bend as FullValues
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_panel_product_placement.bend as PanelPlacement
import ../../lib/proofs/tensor_panel_product_samples.bend as Panels
import ../../lib/proofs/tensor_panel_product_tasks.bend as PanelTasks
import ../../lib/proofs/tensor_pieces_model.bend as Model
import ../../lib/proofs/tensor_point_product_evaluation.bend as Evaluate
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_placement.bend as Placed
import ../../lib/proofs/tensor_point_product_result.bend as Result
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_point_product_tasks.bend as Shared
import ../../lib/proofs/tensor_point_product_values.bend as Values
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/kernel_gemm_point_jobs.bend as Jobs
import ../../lib/proofs/tensor_shape_refinement.bend as Shape
import ../../lib/proofs/tensor_shared_product_placement.bend as SharedPlacement
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/tensor_tiled_padded_cells.bend as Padded
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_tasks.bend as Tasks
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_tiled_map.bend as TiledMap
import ../../lib/proofs/kernel_gemm_output_groups.bend as Groups
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_grid_pieces.bend as GridLaws
import ../../lib/proofs/tensor_grid_ranges.bend as GridRanges
import ../../lib/proofs/tensor_padded_pieces.bend as PiecePadding
import ../../lib/proofs/tensor_point_product_operands.bend as Operands
import ../../lib/proofs/tensor_storage_count.bend as Count
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/tensor_matrix_products.bend as Matrix
import ../../lib/proofs/kernel_gemm_point_refinement.bend as PointLaws
import ../../lib/proofs/tensor_cell_room.bend as CellRoom
import ../../lib/proofs/tensor_tiled_relane.bend as LaneLayout
import ./shared_point_product_contract.bend as SharedContract
import ./panel_point_product_lanes.bend as Lanes
import ./panel_point_product_result.bend as Leaves
import ./product_values.bend as Logical

# ---- A bias of +0 ----

def zero_cell(+geometry: Cells.Cells,+index: U32) ->
  {Representation.lifted(~F32,~FullValues.constant(~F32),Representation.Lifted{0.0,geometry},index) == Products.zero() : Tile.Tile8}:
  Equal.sym(Tile.Tile8,Products.zero(),Representation.lifted(~F32,~FullValues.constant(~F32),Representation.Lifted{0.0,geometry},index),
    FullValues.zero_agree(Products.zero(),Representation.Lifted{0.0,geometry},Gemm.Both{{==},{==}},index))

type ZeroBias is Data:
  ZeroBias{geometry: Cells.Cells,content: Buffer.Content<Tile.Tile8>}

def LiftedAt(+source: ZeroBias,+index: Nat) -> Data:
  ZeroBias{+geometry,+content} = source
  Contract.At(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,geometry},content,index)

def ZeroAt(+source: ZeroBias,+index: Nat) -> Data:
  ZeroBias{+geometry,+content} = source
  Contract.At(~Tile.Tile8,~Unit,~Lanes.zero_tile,Unit{},content,index)

def zero_at(+source: ZeroBias,+index: Nat,+lifted: LiftedAt(source,index)) -> ZeroAt(source,index):
  ZeroBias{+geometry,+content} = source
  +word = U32.from_nat(index)
  Equal.trans(Tile.Tile8,Buffer.value(~Tile.Tile8,word,content),
    Representation.lifted(~F32,~FullValues.constant(~F32),Representation.Lifted{0.0,geometry},word),Products.zero(),lifted,zero_cell(geometry,word))

def zero_bias(+geometry: Cells.Cells,+channels: U32,+content: Buffer.Content<Tile.Tile8>,
  +count: {Buffer.content_length(~Tile.Tile8,content) == U32.to_nat(Gemm.blocks(channels)) : Nat},
  values: Contract.Values(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,geometry},content)) ->
  Lanes.Bias<content,channels>:
  Lanes.Bias{count,
    Members.transform_all(~Nat,~ZeroBias,~LiftedAt,~ZeroAt,~zero_at,Ranges.range(Buffer.content_length(~Tile.Tile8,content),0n),ZeroBias{geometry,content},values)}

# ---- The window of one image ----

def RightWindowAt(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+images: U32,+inner: U32,+channels: U32,+image: U32,
  +holding: Held.Holding<Tile.Tile8>,+length: U32) -> Data:
  Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Right>,Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},holding,(image * length : U32),((image * length : U32) + length : U32),
    Shape.elements([images,inner,Gemm.blocks(channels)])>

def LeftWindowAt(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+images: U32,+rows: U32,+inner: U32,+image: U32,
  +holding: Held.Holding<Tile.Tile8>,+length: U32,+count: U32) -> Data:
  Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Left>,Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},holding,(image * length : U32),((image * length : U32) + (count * length : U32) : U32),
    Shape.elements([images,rows,Gemm.blocks(inner)])>

def RightSpanAt(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+images: U32,+inner: U32,+channels: U32,+image: U32,
  +holding: Held.Holding<Tile.Tile8>,+length: U32,+count: U32) -> Data:
  Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Right>,Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},holding,(image * length : U32),((image * length : U32) + (count * length : U32) : U32),
    Shape.elements([images,inner,Gemm.blocks(channels)])>

def left_window(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+images: U32,+rows: U32,+inner: U32,+image: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+one: {count == 1 : U32},
  +window: LeftWindowAt(~Left,~left_value,left,images,rows,inner,image,holding,Panels.cell_width(rows,inner),count)) ->
  Panels.ImageWindow(~Left,~left_value,left,images,rows,inner,image,holding):
  +length = Panels.cell_width(rows,inner)
  +start = (image * length : U32)
  %Samples.one_times(length) :
    Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Left>,Representation.lifted(~Left,~left_value),
      Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},holding,start,(start + _ : U32),Shape.elements([images,rows,Gemm.blocks(inner)])>
  %one : LeftWindowAt(~Left,~left_value,left,images,rows,inner,image,holding,length,_)
  window

def right_window(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+images: U32,+inner: U32,+channels: U32,+image: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+one: {count == 1 : U32},
  +window: RightSpanAt(~Right,~right_value,right,images,inner,channels,image,holding,(Gemm.blocks(channels) * inner : U32),count)) ->
  Panels.ImageWindow(~Right,~right_value,right,images,inner,channels,image,holding):
  +length = (Gemm.blocks(channels) * inner : U32)
  +start = (image * length : U32)
  %Arithmetic.multiply_commutative(Gemm.blocks(channels),inner) : RightWindowAt(~Right,~right_value,right,images,inner,channels,image,holding,_)
  %Samples.one_times(length) :
    Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Right>,Representation.lifted(~Right,~right_value),
      Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},holding,start,(start + _ : U32),Shape.elements([images,inner,Gemm.blocks(channels)])>
  %one : RightSpanAt(~Right,~right_value,right,images,inner,channels,image,holding,length,_)
  window

# ---- Leaves and their tree ----

def target_of(~Left: Data,~Right: Data,+left: Left,+right: Right,+images: U32,+rows: U32,+inner: U32,+channels: U32) ->
  Representation.Lifted<Logical.Batched<Left,Right>>:
  Representation.Lifted{Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},Leaves.output_geometry(images,rows,channels)}

def CheckedPanel(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+padding: Padded.Layout,
  +low: Nat,+high: Nat,+task: Shared.TaskModel()) -> Data:
  Result.Checked<Representation.Lifted<Logical.Batched<Left,Right>>,Representation.lifted(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value)),
    target_of(~Left,~Right,left,right,images,rows,inner,channels),Padded.layout_padding(padding),limits,Pieces.ranges_of_grid(grid),low,high,task>

# What every leaf shares: the operands, their room, the output layout and the
# bias of +0.
type Common<-Ready: U32 -> Data,-limits: Machine.Limits,-images: U32,-rows: U32,-inner: U32,-channels: U32,-grid: Grid.Grid,-padding: Padded.Layout,
  -origin: Buffer.Content<Tile.Tile8>> is Data:
  Common{dims: PanelPlacement.Dimensions<limits,rows,channels>,layout: PanelPlacement.Layout<limits,Products.Shape{rows,inner,channels,grid},images>,
    domain: Lanes.Domain<limits,images,rows,inner,channels>,size: Panels.Size<limits,images,rows,channels>,
    geometry: {Padded.layout_geometry(padding) == Leaves.output_geometry(images,rows,channels) : Cells.Cells},
    total: {Padded.layout_total(padding) == Panels.physical_size(images,rows,channels) : Nat},
    ready: Ready(inner),bias: Lanes.Bias<origin,channels>}

def common_dims(~Ready: U32 -> Data,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+padding: Padded.Layout,
  +origin: Buffer.Content<Tile.Tile8>,+common: Common<Ready,limits,images,rows,inner,channels,grid,padding,origin>) -> PanelPlacement.Dimensions<limits,rows,channels>:
  Common{+dims,layout,domain,size,geometry,total,ready,bias} = common
  dims

def common_layout(~Ready: U32 -> Data,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+padding: Padded.Layout,
  +origin: Buffer.Content<Tile.Tile8>,+common: Common<Ready,limits,images,rows,inner,channels,grid,padding,origin>) ->
  PanelPlacement.Layout<limits,Products.Shape{rows,inner,channels,grid},images>:
  Common{dims,+layout,domain,size,geometry,total,ready,bias} = common
  layout

def LeftOk(~Left: Data,~left_value: Left -> U32 -> F32,+left: Left,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +tree: PanelTasks.OperandTree()) -> Data:
  Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),~Products.Shape,~Products.panel_left,
    Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},Products.Shape{rows,inner,channels,grid},Shape.elements([images,rows,Gemm.blocks(inner)]),
    PanelTasks.operand_ranges(tree),PanelTasks.operand_windows(~PanelTasks.input_holding,tree))

def RightOk(~Right: Data,~right_value: Right -> U32 -> F32,+right: Right,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +tree: PanelTasks.OperandTree()) -> Data:
  Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),~Products.Shape,~Products.panel_right,
    Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},Products.Shape{rows,inner,channels,grid},Shape.elements([images,inner,Gemm.blocks(channels)]),
    PanelTasks.operand_ranges(tree),PanelTasks.operand_windows(~PanelTasks.weight_holding,tree))

def image_inside(+image: U32,+count: U32,+images: U32,+one: {count == 1 : U32},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(images)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(image),U32.to_nat(images)) == True{} : Bool}:
  +first = U32.to_nat(image)
  Order.succ_le_lt(first,U32.to_nat(images),
    %Panels.one_more(first) : {Nat.is_le(_,U32.to_nat(images)) == True{} : Bool}
    %one : {Nat.is_le(Nat.add(first,U32.to_nat(_)),U32.to_nat(images)) == True{} : Bool}
    inside)

def leaf_checked(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+padding: Padded.Layout,
  +origin: Buffer.Content<Tile.Tile8>,+common: Common<Ready,limits,images,rows,inner,channels,grid,padding,origin>,
  +image: U32,+count: U32,+operands: PanelTasks.Operands,+result: Shared.TaskModel(),+one: {count == 1 : U32},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(images)) == True{} : Bool},
  +facts: PanelPlacement.LeafLayout<limits,Products.Shape{rows,inner,channels,grid},image,count>,
  +emitted: PanelTasks.Emitted<run,Products.Shape{rows,inner,channels,grid},image,operands,result,origin>,
  +left_ok: LeftWindowAt(~Left,~left_value,left,images,rows,inner,image,PanelTasks.input_holding(operands),Panels.cell_width(rows,inner),count),
  +right_ok: RightSpanAt(~Right,~right_value,right,images,inner,channels,image,PanelTasks.weight_holding(operands),(Gemm.blocks(channels) * inner : U32),count)) ->
  CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,
    Nat.mul(U32.to_nat(image),U32.to_nat(PanelTasks.image_width(Products.Shape{rows,inner,channels,grid}))),
    Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(PanelTasks.image_width(Products.Shape{rows,inner,channels,grid}))),result):
  Common{+dims,+layout,+domain,+size,+geometry_exact,+total,+ready,+bias_values} = common
  PanelTasks.Operands{+input,+weights} = operands
  PanelPlacement.LeafLayout{+first_exact,+width_exact,+last_exact,+space,+laid} = facts
  PanelTasks.Emitted{+bias,+contents,+same_bias} = emitted
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.panel_geometry(work,image)
  +stage = Point.panel_stage_model(~run,work,image,Held.pack(~Tile.Tile8,input),Held.pack(~Tile.Tile8,weights),bias)
  +plan = {Placed.RectanglePlan{rectangle,stage,result,Fragments.rectangle_start(rectangle),PanelTasks.image_width(work),{==},width_exact,first_exact,last_exact,space,
    Equal.trans(Model.Content<Tile.Tile8>,Model.content(~Tile.Tile8,Placed.task_pieces(result)),Point.task_contents(Shared.pack_task(result)),Point.rectangle_model(rectangle,stage),
      Equal.sym(Model.Content<Tile.Tile8>,Point.task_contents(Shared.pack_task(result)),Model.content(~Tile.Tile8,Placed.task_pieces(result)),Placed.task_contents(result)),contents),laid} :
    Placed.Plan<limits,Pieces.ranges_of_grid(grid),Nat.mul(U32.to_nat(image),U32.to_nat(PanelTasks.image_width(work))),
      Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(PanelTasks.image_width(work)))>}
  +own_bias = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,bias),origin,same_bias) : Lanes.Bias<_,channels> bias_values :
    Lanes.Bias<Buffer.content(~Tile.Tile8,bias),channels>}
  +leaf = {Leaves.PanelWindow{limits,left,right,images,rows,inner,channels,grid,image,input,weights,bias,domain,image_inside(image,count,images,one,inside),
    left_window(~Left,~left_value,left,images,rows,inner,image,count,input,one,left_ok),
    right_window(~Right,~right_value,right,images,inner,channels,image,count,weights,one,right_ok),
    own_bias,ready,Leaves.OutputLayout{size,padding,geometry_exact,total},space} : Leaves.PanelWindow<Ready,Left,left_value,Right,right_value>}
  Result.Checked{plan,{==},Leaves.leaf_values(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,leaf)}

def checked_tree(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+padding: Padded.Layout,
  +origin: Buffer.Content<Tile.Tile8>,+common: Common<Ready,limits,images,rows,inner,channels,grid,padding,origin>,
  tree: PanelTasks.OperandTree(),output: Shared.TaskModel(),
  +runs: Runs.Run(PanelTasks.operand_ranges(tree),1n,U32.to_nat(images)),+ones: PanelTasks.Singletons(PanelTasks.operand_ranges(tree)),
  evidence: PanelTasks.Emissions(~run,Products.Shape{rows,inner,channels,grid},tree,output,origin),
  left_ok: LeftOk(~Left,~left_value,left,images,rows,inner,channels,grid,tree),right_ok: RightOk(~Right,~right_value,right,images,rows,inner,channels,grid,tree)) ->
  CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,
    PanelPlacement.coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_low(PanelTasks.operand_ranges(tree))),
    PanelPlacement.coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_high(PanelTasks.operand_ranges(tree))),output):
  match tree output:
    case Partition.Leaf{+operands,+first,+count,+details} +result:
      Held.Both{left_first,Held.Both{left_count,+left_window}} = left_ok
      Held.Both{right_first,Held.Both{right_count,+right_window}} = right_ok
      +inside = Runs.fits_inside(1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(images),runs)
      leaf_checked(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,origin,
        common,first,count,operands,result,ones,inside,
        PanelPlacement.leaf_layout(limits,images,rows,inner,channels,grid,common_dims(~Ready,limits,images,rows,inner,channels,grid,padding,origin,common),
          common_layout(~Ready,limits,images,rows,inner,channels,grid,padding,origin,common),first,count,ones,inside),evidence,left_window,right_window)
    case Partition.Fork{+lower,+upper} Partition.Fork{+left_output,+right_output}:
      Held.Both{+left_run,+rest} = runs
      Held.Both{+right_run,+glue} = rest
      Held.Both{+left_ones,+right_ones} = ones
      Held.Both{+left_emitted,+right_emitted} = evidence
      Held.Both{+left_reads,+right_reads} = left_ok
      Held.Both{+left_weights,+right_weights} = right_ok
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      +left_ranges = PanelTasks.operand_ranges(lower)
      +right_ranges = PanelTasks.operand_ranges(upper)
      +bound = U32.to_nat(images)
      +width = U32.to_nat(PanelTasks.image_width(work))
      +low = PanelPlacement.coordinate(work,Runs.range_low(left_ranges))
      +middle = PanelPlacement.coordinate(work,Runs.range_high(left_ranges))
      +second = PanelPlacement.coordinate(work,Runs.range_low(right_ranges))
      +high = PanelPlacement.coordinate(work,Runs.range_high(right_ranges))
      +edge = Equal.sym(Nat,middle,second,Equal.cong(Nat,Nat,number => PanelPlacement.coordinate(work,number),Runs.range_high(left_ranges),Runs.range_low(right_ranges),glue))
      Result.checked_join(~Representation.Lifted<Logical.Batched<Left,Right>>,~Representation.lifted(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value)),
        target_of(~Left,~Right,left,right,images,rows,inner,channels),Padded.layout_padding(padding),limits,Pieces.ranges_of_grid(grid),low,middle,high,left_output,right_output,
        Index.mul_le(Runs.range_low(left_ranges),Runs.range_high(left_ranges),width,
          SharedPlacement.run_order(left_ranges,1n,bound,Runs.run_fits(left_ranges,1n,bound,left_run))),
        %edge : {Nat.is_le(_,high) == True{} : Bool}
        Index.mul_le(Runs.range_low(right_ranges),Runs.range_high(right_ranges),width,
          SharedPlacement.run_order(right_ranges,1n,bound,Runs.run_fits(right_ranges,1n,bound,right_run))),
        checked_tree(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,origin,common,
          lower,left_output,left_run,left_ones,left_emitted,left_reads,left_weights),
        %edge : CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,_,high,right_output)
        checked_tree(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,origin,common,
          upper,right_output,right_run,right_ones,right_emitted,right_reads,right_weights))
    case Partition.Fork{+lower,+upper} Partition.Leaf{+pieces,+first,+count,+detail}:
      Empty.absurd(CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,
        PanelPlacement.coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_low(PanelTasks.operand_ranges(lower))),
        PanelPlacement.coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_high(PanelTasks.operand_ranges(upper))),
        Partition.Leaf{pieces,first,count,detail}),evidence)

# ---- Sizes ----

# The extents and rooms that the guard and the operand contracts give.
type PanelSizes<-limits: Machine.Limits,-images: U32,-rows: U32,-inner: U32,-channels: U32> is Data:
  PanelSizes{default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
    images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},
    images_room: {Nat.is_le(U32.to_nat(images),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
    inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
    channels_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    input_room: {Nat.is_le(Shape.elements([images,rows,Gemm.blocks(inner)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weights_room: {Nat.is_le(Shape.elements([images,inner,Gemm.blocks(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Shape.elements([images,rows,Gemm.blocks(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    logical_room: {Nat.is_le(Shape.elements([images,rows,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    right_room: {Nat.is_le(Shape.elements([images,inner,channels]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def three_positive(+a: U32,+b: U32,+c: U32,+a_positive: {Nat.is_lt(0n,U32.to_nat(a)) == True{} : Bool},+b_positive: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool},
  +c_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}) -> {Nat.is_lt(0n,Shape.elements([a,b,c])) == True{} : Bool}:
  +product = Nat.mul(U32.to_nat(a),Nat.mul(U32.to_nat(b),U32.to_nat(c)))
  %Equal.sym(Nat,Shape.elements([a,b,c]),product,Shape.elements_three(a,b,c)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(a),Nat.mul(U32.to_nat(b),U32.to_nat(c)),a_positive,Order.product_positive(U32.to_nat(b),U32.to_nat(c),b_positive,c_positive))

# The product of the last two extents fits when the three do.
def tail_room(+limits: Machine.Limits,+a: U32,+b: U32,+c: U32,+a_positive: {Nat.is_lt(0n,U32.to_nat(a)) == True{} : Bool},
  +room: {Nat.is_le(Shape.elements([a,b,c]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(b),U32.to_nat(c)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +tail = Nat.mul(U32.to_nat(b),U32.to_nat(c))
  +whole = Nat.mul(U32.to_nat(a),tail)
  +cap = U32.to_nat(Machine.capacity(limits))
  Order.transitive(tail,whole,cap,
    %Index.multiply_commutative(tail,U32.to_nat(a)) : {Nat.is_le(tail,_) == True{} : Bool} Index.product_covers(tail,U32.to_nat(a),a_positive),
    %Shape.elements_three(a,b,c) : {Nat.is_le(_,cap) == True{} : Bool} room)

def sized(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+width: U32,+same: {width == Panels.cell_width(rows,channels) : U32},
  +size: Panels.Size<limits,images,rows,channels>,+room: {Nat.is_le(Shape.elements([images,rows,Gemm.blocks(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,Shape.elements([images,rows,Gemm.blocks(channels)])) == True{} : Bool}) ->
  PanelTasks.SizedPanel<limits,images,[images,rows,Gemm.blocks(channels)],width>:
  Panels.Size{+width_exact,outer,image_room,whole_room} = size
  +i = U32.to_nat(images)
  +cell = Panels.cell_width(rows,channels)
  +elements = Shape.elements([images,rows,Gemm.blocks(channels)])
  +exact = Equal.trans(Nat,elements,Nat.mul(i,Panels.image_size(rows,channels)),Nat.mul(i,U32.to_nat(cell)),Shape.elements_three(images,rows,Gemm.blocks(channels)),
    Equal.cong(Nat,Nat,value => Nat.mul(i,value),Panels.image_size(rows,channels),U32.to_nat(cell),Equal.sym(Nat,U32.to_nat(cell),Panels.image_size(rows,channels),width_exact)))
  PanelTasks.SizedPanel{%Equal.sym(U32,width,cell,same) : {elements == Nat.mul(i,U32.to_nat(_)) : Nat} exact,room,nonempty}

def size_room(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+size: Panels.Size<limits,images,rows,channels>) ->
  {Nat.is_le(Panels.image_size(rows,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Panels.Size{width,outer,+image_room,room} = size
  image_room

def size_width(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+size: Panels.Size<limits,images,rows,channels>) ->
  {U32.to_nat(Panels.cell_width(rows,channels)) == Panels.image_size(rows,channels) : Nat}:
  Panels.Size{+width,outer,image_room,room} = size
  width

def budget_room(+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  {Nat.is_le(U32.to_nat(Gemm.block_tiles()),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +cap = Machine.capacity(limits)
  Bounds.as_nat_le(Gemm.block_tiles(),cap,%Equal.sym(U32,cap,Machine.capacity(Default.limits()),default_capacity) : {U32.is_le(Gemm.block_tiles(),_) == True{} : Bool} {==})

# ---- The output layout ----

def output_shape(+images: U32,+rows: U32,+channels: U32) -> List<&2,U32>:
  [images,rows,Gemm.blocks(channels)]

def output_layout(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,
  +outer: {U32.to_nat((images * rows : U32)) == Panels.row_total(images,rows) : Nat},
  +room: {Nat.is_le(Nat.mul(Panels.row_total(images,rows),U32.to_nat(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Padded.Layout:
  +i = U32.to_nat(images)
  +r = U32.to_nat(rows)
  +g = U32.to_nat(Gemm.blocks(channels))
  +c = U32.to_nat(channels)
  +total = Panels.row_total(images,rows)
  +outer_word = (images * rows : U32)
  +elements = Shape.elements(output_shape(images,rows,channels))
  +counted = {%Equal.sym(Nat,U32.to_nat(outer_word),total,outer) : {Nat.is_le(Nat.mul(c,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Index.multiply_commutative(total,c) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
    {Nat.is_le(Nat.mul(c,U32.to_nat(outer_word)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells = {%Equal.sym(Nat,U32.to_nat(outer_word),total,outer) : {Nat.mul(g,_) == elements : Nat}
    Equal.trans(Nat,Nat.mul(g,total),Nat.mul(total,g),elements,Index.multiply_commutative(g,total),
      Equal.trans(Nat,Nat.mul(total,g),Nat.mul(i,Nat.mul(r,g)),elements,Index.assoc(i,r,g),
        Equal.sym(Nat,elements,Nat.mul(i,Nat.mul(r,g)),Shape.elements_three(images,rows,Gemm.blocks(channels))))) :
    {Nat.mul(g,U32.to_nat(outer_word)) == elements : Nat}}
  Padded.Layout{limits,channels,outer_word,Grid.Innermost{},outer_word,1,Grid.Innermost{},elements,Public.Cover{counted,cells,cells,Public.Along{{==},{==}}}}

def grid_low(+cells: List<&2,U32>,+target: U32,+unit: Nat,+layout: GridRanges.GridLayout<cells,target,unit>) ->
  {Runs.range_low(Pieces.ranges_of_grid(Grid.grid(cells,target))) == 0n : Nat}:
  GridRanges.GridLayout{run,+low,high} = layout
  low

def grid_high(+cells: List<&2,U32>,+target: U32,+unit: Nat,+layout: GridRanges.GridLayout<cells,target,unit>) ->
  {Runs.range_high(Pieces.ranges_of_grid(Grid.grid(cells,target))) == Shape.elements(cells) : Nat}:
  GridRanges.GridLayout{run,low,+high} = layout
  high

def PanelResultWith(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8),+padding: TiledMap.Padding) -> Data:
  Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},[images,rows,channels],
    TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{pieces,[images,rows,Gemm.blocks(channels)],Products.zero(),Tensors.Usable{},workers},2n,channels,Grid.Innermost{}},padding)))

def PanelResult(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Data:
  PanelResultWith(~Left,~left_value,~Right,~right_value,left,right,images,rows,inner,channels,workers,pieces,
    TiledMap.Padding{channels,Gemm.blocks(channels),(images * rows : U32),Grid.Innermost{}})

def CheckedOver(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+padding: Padded.Layout,
  +ranges: Pieces.Ranges,+task: Shared.TaskModel()) -> Data:
  +work = PanelTasks.work_of(images,rows,inner,channels)
  CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,PanelTasks.panel_grid(work),padding,
    PanelPlacement.coordinate(work,Runs.range_low(ranges)),PanelPlacement.coordinate(work,Runs.range_high(ranges)),task)

def seal_checked(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,+padding: Padded.Layout,
  +nonempty: {Nat.is_lt(0n,Padded.layout_total(padding)) == True{} : Bool},+same_limits: {PiecePadding.layout_limits(padding) == limits : Machine.Limits},
  +total: {Padded.layout_total(padding) == Shape.elements(output_shape(images,rows,channels)) : Nat},
  +counted: Global.Counted<[images,rows,channels]>,
  +room: {Nat.is_le(Shape.elements(output_shape(images,rows,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +layout: GridRanges.GridLayout<output_shape(images,rows,channels),Grid.piece_target(),1n>,+task: Shared.TaskModel(),
  +checked: CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,Tensors.grid_of(output_shape(images,rows,channels)),padding,
    Runs.range_low(Pieces.ranges_of_grid(Tensors.grid_of(output_shape(images,rows,channels)))),Runs.range_high(Pieces.ranges_of_grid(Tensors.grid_of(output_shape(images,rows,channels)))),task),
  pieces: Pieces.Pieces(~Tile.Tile8),
  +placed: {pieces == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(Tensors.grid_of(output_shape(images,rows,channels))),Products.zero(),Shared.pack_task(task)) : Pieces.Pieces(~Tile.Tile8)}) ->
  PanelResultWith(~Left,~left_value,~Right,~right_value,left,right,images,rows,inner,channels,workers,pieces,Padded.layout_padding(padding)):
  Result.Checked{+plan,+same,+values} = checked
  +logical = {Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)} : Logical.Batched<Left,Right>}
  +shape = output_shape(images,rows,channels)
  +dims = {[images,rows,channels] : List<&2,U32>}
  +ranges = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +first = Runs.range_low(ranges)
  +last = Runs.range_high(ranges)
  +geometry = Public.geometry_of(dims,2n,Grid.Innermost{})
  Result.plan_contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    logical,dims,shape,limits,2n,channels,Grid.Innermost{},workers,padding,nonempty,same_limits,total,{==},
    Public.Laid{{==},{==},{==},counted,{==}},room,default_capacity,1n,layout,plan,
    %Equal.sym(Cells.Cells,geometry,Leaves.output_geometry(images,rows,channels),Samples.matrix_geometry_of(images,rows,channels,Grid.Innermost{})) :
      Result.Masked(~Representation.Lifted<Logical.Batched<Left,Right>>,~Representation.lifted(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value)),
        Representation.Lifted{logical,_},Padded.layout_padding(padding),limits,ranges,first,last,plan)
    values,pieces,
    %Equal.sym(Shared.TaskModel(),Placed.planned_tasks(limits,ranges,first,last,plan),task,same) :
      {pieces == Tensors.placed(~Tile.Tile8,ranges,Products.zero(),Shared.pack_task(_)) : Pieces.Pieces(~Tile.Tile8)}
    placed)

def image_runs(+ranges: Pieces.Ranges,+images: U32,+facts: PanelTasks.ImageRun(ranges,0,images,U32.to_nat(images))) -> Runs.Run(ranges,1n,U32.to_nat(images)):
  Held.Both{Held.Both{+run,same},ones} = facts
  run

def image_low(+ranges: Pieces.Ranges,+images: U32,+facts: PanelTasks.ImageRun(ranges,0,images,U32.to_nat(images))) ->
  {Runs.range_low(ranges) == Runs.range_low(Pieces.Range{0,images}) : Nat}:
  Held.Both{Held.Both{run,Held.Both{+low,high}},ones} = facts
  low

def image_high(+ranges: Pieces.Ranges,+images: U32,+facts: PanelTasks.ImageRun(ranges,0,images,U32.to_nat(images))) ->
  {Runs.range_high(ranges) == Runs.range_high(Pieces.Range{0,images}) : Nat}:
  Held.Both{Held.Both{run,Held.Both{low,+high}},ones} = facts
  high

def image_ones(+ranges: Pieces.Ranges,+images: U32,+facts: PanelTasks.ImageRun(ranges,0,images,U32.to_nat(images))) -> PanelTasks.Singletons(ranges):
  Held.Both{run,+ones} = facts
  ones

def panel_exact(+limits: Machine.Limits,+images: U32,+shape: List<&2,U32>,+width: U32,+sized: PanelTasks.SizedPanel<limits,images,shape,width>) ->
  {Shape.elements(shape) == Nat.mul(U32.to_nat(images),U32.to_nat(width)) : Nat}:
  PanelTasks.SizedPanel{+exact,room,nonempty} = sized
  exact

def seal_output(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +sizes: PanelSizes<limits,images,rows,inner,channels>,+ready: Ready(inner),+origin: Buffer.Content<Tile.Tile8>,+bias_values: Lanes.Bias<origin,channels>,
  -left_pieces: Pieces.Pieces(~Tile.Tile8),-right_pieces: Pieces.Pieces(~Tile.Tile8),native: Pieces.Pieces(~Tile.Tile8),
  checked: PanelTasks.PanelChecked<run,Representation.Lifted<Left>,Representation.lifted(~Left,~left_value),Representation.Lifted<Right>,Representation.lifted(~Right,~right_value),
    Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},
    images,rows,inner,channels,left_pieces,right_pieces,native,origin>) ->
  PanelResult(~Left,~left_value,~Right,~right_value,left,right,images,rows,inner,channels,workers,native):
  PanelSizes{+default_capacity,+images_positive,+images_room,+rows_positive,+rows_room,+inner_positive,+inner_room,+channels_positive,+channels_room,
    +input_room,+weights_room,+output_room,+logical_room,+right_room} = sizes
  PanelTasks.PanelChecked{+left_reads,+right_reads,+output} = checked
  Windows.Windowed{+left_windows,left_packed,+left_ok} = left_reads
  Windows.Windowed{+right_windows,right_packed,+right_ok} = right_reads
  PanelTasks.PanelOutput{+model,+placed,writes,+tree,+arranged,+emitted,+input_windows,+weight_windows} = output
  +work = PanelTasks.work_of(images,rows,inner,channels)
  +shape = output_shape(images,rows,channels)
  +grid = Tensors.grid_of(shape)
  +ranges = Tasks.component_ranges(images)
  +target = Grid.piece_target()
  +cap = U32.to_nat(Machine.capacity(limits))
  +left_source = {Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)} : Representation.Lifted<Left>}
  +right_source = {Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)} : Representation.Lifted<Right>}
  +channel_groups = Groups.groups_positive(limits,channels,channels_room,channels_positive)
  +left_size = Panels.size(limits,images,rows,inner,images_positive,Groups.groups_positive(limits,inner,inner_room,inner_positive),input_room)
  +right_size = Panels.size(limits,images,inner,channels,images_positive,channel_groups,weights_room)
  +out_size = Panels.size(limits,images,rows,channels,images_positive,channel_groups,output_room)
  +out_nonempty = three_positive(images,rows,Gemm.blocks(channels),images_positive,rows_positive,channel_groups)
  +output_exact = panel_exact(limits,images,shape,PanelTasks.image_width(work),
    sized(limits,images,rows,channels,PanelTasks.image_width(work),{==},out_size,output_room,out_nonempty))
  +width = U32.to_nat(PanelTasks.image_width(work))
  +full = Nat.mul(U32.to_nat(images),width)
  +counted = GridRanges.grid_count(limits,shape,target,output_room,out_nonempty)
  +last = GridRanges.last_extent(shape)
  +divides = {GridRanges.Multiple{last,Equal.sym(Nat,Nat.mul(last,1n),last,Order.times_one(last))} : GridRanges.Multiple<last,1n>}
  +out_layout = GridRanges.grid_layout(limits,shape,target,output_room,out_nonempty,1n,divides)
  +cells_total = GridLaws.grid_cells(grid)
  +layout = {PanelPlacement.Layout{GridRanges.grid_facts(limits,shape,target,output_room,out_nonempty),
      Equal.trans(Nat,cells_total,Shape.elements(shape),full,counted,output_exact),
      size_width(limits,images,rows,channels,out_size),
      DivisionOrder.right_factor_positive(U32.to_nat(images),width,%output_exact : {Nat.is_lt(0n,_) == True{} : Bool} out_nonempty),
      %Equal.sym(Nat,cells_total,Shape.elements(shape),counted) : {Nat.is_le(_,cap) == True{} : Bool} output_room,
      %Equal.sym(Nat,cells_total,Shape.elements(shape),counted) : Runs.Run(Pieces.ranges_of_grid(grid),1n,_) GridRanges.layout_run(shape,target,1n,out_layout)} :
    PanelPlacement.Layout<limits,Products.Shape{rows,inner,channels,grid},images>}
  +i = U32.to_nat(images)
  +r = U32.to_nat(rows)
  +c = U32.to_nat(channels)
  +output_rows = {%Equal.sym(Nat,Nat.mul(Nat.mul(i,r),c),Nat.mul(i,Nat.mul(r,c)),Index.assoc(i,r,c)) : {Nat.is_le(_,cap) == True{} : Bool}
    %Shape.elements_three(images,rows,channels) : {Nat.is_le(_,cap) == True{} : Bool} logical_room :
    {Nat.is_le(Nat.mul(Panels.row_total(images,rows),c),cap) == True{} : Bool}}
  +domain = {Lanes.Domain{rows_positive,channels_positive,rows_room,channels_room,left_size,right_size,
    %Index.multiply_commutative(U32.to_nat(inner),c) : {Nat.is_le(_,cap) == True{} : Bool} tail_room(limits,images,inner,channels,images_positive,right_room),
    output_rows,Values.block_span(limits,inner,inner_room,inner_positive,budget_room(limits,default_capacity)),TiledValues.grouped(limits,inner,inner_room),
    Operands.PairSpace{Groups.groups_fit(limits,channels,channels_room),inner_room,
      %Index.multiply_commutative(U32.to_nat(inner),U32.to_nat(Gemm.blocks(channels))) : {Nat.is_le(_,cap) == True{} : Bool}
      tail_room(limits,images,inner,Gemm.blocks(channels),images_positive,weights_room)},
    size_room(limits,images,rows,channels,out_size)} : Lanes.Domain<limits,images,rows,inner,channels>}
  +padding = output_layout(limits,images,rows,channels,Leaves.size_outer(limits,images,rows,channels,out_size),output_rows)
  +common = {Common{PanelPlacement.Dimensions{rows_room,rows_positive,channels_room,channels_positive},layout,domain,out_size,{==},
    Panels.cell_count(images,rows,channels),ready,bias_values} : Common<Ready,limits,images,rows,inner,channels,grid,padding,origin>}
  +image_run = PanelTasks.components(limits,images,images_positive,images_room)
  +tree_ranges = PanelTasks.operand_ranges(tree)
  +back = Equal.sym(Pieces.Ranges,tree_ranges,ranges,arranged)
  +tree_runs = {%back : Runs.Run(_,1n,i) image_runs(ranges,images,image_run) : Runs.Run(tree_ranges,1n,i)}
  +tree_ones = {%back : PanelTasks.Singletons(_) image_ones(ranges,images,image_run) : PanelTasks.Singletons(tree_ranges)}
  +left_tree = {%back : Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),~Products.Shape,~Products.panel_left,
      left_source,work,Shape.elements([images,rows,Gemm.blocks(inner)]),_,PanelTasks.operand_windows(~PanelTasks.input_holding,tree))
    %Equal.sym(Windows.WindowModel<Tile.Tile8>,PanelTasks.operand_windows(~PanelTasks.input_holding,tree),left_windows,input_windows) :
      Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),~Products.Shape,~Products.panel_left,
        left_source,work,Shape.elements([images,rows,Gemm.blocks(inner)]),ranges,_)
    left_ok : LeftOk(~Left,~left_value,left,images,rows,inner,channels,grid,tree)}
  +right_tree = {%back : Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),~Products.Shape,~Products.panel_right,
      right_source,work,Shape.elements([images,inner,Gemm.blocks(channels)]),_,PanelTasks.operand_windows(~PanelTasks.weight_holding,tree))
    %Equal.sym(Windows.WindowModel<Tile.Tile8>,PanelTasks.operand_windows(~PanelTasks.weight_holding,tree),right_windows,weight_windows) :
      Windows.WindowsOk(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),~Products.Shape,~Products.panel_right,
        right_source,work,Shape.elements([images,inner,Gemm.blocks(channels)]),ranges,_)
    right_ok : RightOk(~Right,~right_value,right,images,rows,inner,channels,grid,tree)}
  +layout_ranges = Pieces.ranges_of_grid(grid)
  +high_eq = Equal.trans(Nat,full,Shape.elements(shape),Runs.range_high(layout_ranges),Equal.sym(Nat,Shape.elements(shape),full,output_exact),
    Equal.sym(Nat,Runs.range_high(layout_ranges),Shape.elements(shape),grid_high(shape,target,1n,out_layout)))
  seal_checked(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,workers,padding,out_nonempty,{==},{==},
    Count.counted_within(limits,[images,rows,channels],logical_room,default_capacity),output_room,default_capacity,out_layout,model,
    %Equal.sym(Nat,Runs.range_low(layout_ranges),0n,grid_low(shape,target,1n,out_layout)) :
      CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,_,Runs.range_high(layout_ranges),model)
    %high_eq : CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,0n,_,model)
    %image_low(ranges,images,image_run) :
      CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,PanelPlacement.coordinate(work,_),full,model)
    %image_high(ranges,images,image_run) :
      CheckedPanel(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,
        PanelPlacement.coordinate(work,Runs.range_low(ranges)),PanelPlacement.coordinate(work,_),model)
    %arranged : CheckedOver(~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,padding,_,model)
    checked_tree(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,grid,padding,origin,common,
      tree,model,tree_runs,tree_ones,emitted,left_tree,right_tree),
    native,placed)

# The panels of the actual schedule hold the products of every image.
def panel_result(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  ~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +sizes: PanelSizes<limits,images,rows,inner,channels>,+ready: Ready(inner),+bias: Buffer.Model<Tile.Tile8>,
  +bias_values: Lanes.Bias<Buffer.content(~Tile.Tile8,bias),channels>,+left_model: Model.Model<Tile.Tile8>,+right_model: Model.Model<Tile.Tile8>,
  +left_cover: PanelTasks.Cover(~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},PanelTasks.input_shape(images,rows,inner),Model.pack(~Tile.Tile8,left_model)),
  +right_cover: PanelTasks.Cover(~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},PanelTasks.weights_shape(images,inner,channels),Model.pack(~Tile.Tile8,right_model))) ->
  PanelResult(~Left,~left_value,~Right,~right_value,left,right,images,rows,inner,channels,workers,
    Products.panels(~kernel,images,rows,inner,channels,workers,Model.pack(~Tile.Tile8,left_model),Model.pack(~Tile.Tile8,right_model),Buffer.pack(~Tile.Tile8,bias))):
  PanelSizes{+default_capacity,+images_positive,+images_room,+rows_positive,+rows_room,+inner_positive,+inner_room,+channels_positive,+channels_room,
    +input_room,+weights_room,+output_room,+logical_room,+right_room} = sizes
  +work = PanelTasks.work_of(images,rows,inner,channels)
  +inner_groups = Groups.groups_positive(limits,inner,inner_room,inner_positive)
  +channel_groups = Groups.groups_positive(limits,channels,channels_room,channels_positive)
  +left_size = Panels.size(limits,images,rows,inner,images_positive,inner_groups,input_room)
  +right_size = Panels.size(limits,images,inner,channels,images_positive,channel_groups,weights_room)
  +out_size = Panels.size(limits,images,rows,channels,images_positive,channel_groups,output_room)
  seal_output(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,workers,
    PanelSizes{default_capacity,images_positive,images_room,rows_positive,rows_room,inner_positive,inner_room,channels_positive,channels_room,
      input_room,weights_room,output_room,logical_room,right_room},ready,Buffer.content(~Tile.Tile8,bias),bias_values,Model.pack(~Tile.Tile8,left_model),Model.pack(~Tile.Tile8,right_model),
    Products.panels(~kernel,images,rows,inner,channels,workers,Model.pack(~Tile.Tile8,left_model),Model.pack(~Tile.Tile8,right_model),Buffer.pack(~Tile.Tile8,bias)),
    PanelTasks.panel_product_coverage(~kernel,~run,~agrees,~keeps,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
      ~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
      Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},
      limits,images,rows,inner,channels,workers,bias,images_positive,images_room,
      sized(limits,images,rows,inner,PanelTasks.left_width(work),{==},left_size,input_room,three_positive(images,rows,Gemm.blocks(inner),images_positive,rows_positive,inner_groups)),
      sized(limits,images,inner,channels,PanelTasks.right_width(work),Arithmetic.multiply_commutative(Gemm.blocks(channels),inner),right_size,weights_room,
        three_positive(images,inner,Gemm.blocks(channels),images_positive,inner_positive,channel_groups)),
      sized(limits,images,rows,channels,PanelTasks.image_width(work),{==},out_size,output_room,three_positive(images,rows,Gemm.blocks(channels),images_positive,rows_positive,channel_groups)),
      Model.pack(~Tile.Tile8,left_model),Model.pack(~Tile.Tile8,right_model),left_cover,right_cover))

# ---- The products of matrices of every image ----

def head_room(+limits: Machine.Limits,+a: U32,+b: U32,+c: U32,+b_positive: {Nat.is_lt(0n,U32.to_nat(b)) == True{} : Bool},+c_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +room: {Nat.is_le(Shape.elements([a,b,c]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(a),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +tail = Nat.mul(U32.to_nat(b),U32.to_nat(c))
  +cap = U32.to_nat(Machine.capacity(limits))
  Order.transitive(U32.to_nat(a),Nat.mul(U32.to_nat(a),tail),cap,Index.product_covers(U32.to_nat(a),tail,Order.product_positive(U32.to_nat(b),U32.to_nat(c),b_positive,c_positive)),
    %Shape.elements_three(a,b,c) : {Nat.is_le(_,cap) == True{} : Bool} room)

def middle_room(+limits: Machine.Limits,+a: U32,+b: U32,+c: U32,+a_positive: {Nat.is_lt(0n,U32.to_nat(a)) == True{} : Bool},+c_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},
  +room: {Nat.is_le(Shape.elements([a,b,c]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(b),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +tail = Nat.mul(U32.to_nat(b),U32.to_nat(c))
  Order.transitive(U32.to_nat(b),tail,U32.to_nat(Machine.capacity(limits)),Index.product_covers(U32.to_nat(b),U32.to_nat(c),c_positive),
    tail_room(limits,a,b,c,a_positive,room))

def panel_kernel(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,whole: Bool,
  +left: Left,+right: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,
  +sizes: PanelSizes<limits,images,rows,inner,channels>,+exact: {U32.is_zero((inner % 16 : U32)) == whole : Bool},+bias: Buffer.Model<Tile.Tile8>,
  +bias_values: Lanes.Bias<Buffer.content(~Tile.Tile8,bias),channels>,+left_model: Model.Model<Tile.Tile8>,+right_model: Model.Model<Tile.Tile8>,
  +left_cover: PanelTasks.Cover(~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry((images * rows : U32),inner)},PanelTasks.input_shape(images,rows,inner),Model.pack(~Tile.Tile8,left_model)),
  +right_cover: PanelTasks.Cover(~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((images * inner : U32),channels)},PanelTasks.weights_shape(images,inner,channels),Model.pack(~Tile.Tile8,right_model))) ->
  PanelResult(~Left,~left_value,~Right,~right_value,left,right,images,rows,inner,channels,workers,
    Matrix.panel_products(whole,images,rows,inner,channels,workers,Model.pack(~Tile.Tile8,left_model),Model.pack(~Tile.Tile8,right_model),Buffer.pack(~Tile.Tile8,bias))):
  match whole:
    case True{}:
      panel_result(~GemmBuffer.point_keeping,~Jobs.run,~Point.block_whole,~Values.whole_bias,~Jobs.job_accumulators_with(~Gemm.row_times,~Jobs.whole_inner),
        ~Values.whole_frame,~Evaluate.Whole,~Evaluate.whole_order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,workers,
        sizes,exact,bias,bias_values,left_model,right_model,left_cover,right_cover)
    case False{}:
      panel_result(~GemmBuffer.point_keeping_tail,~Jobs.run_tail,~Point.block_tail,~Values.tail_bias,~Jobs.job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced),
        ~Values.tail_frame,~Evaluate.Tail,~Evaluate.tail_order,~Left,~left_value,~Right,~right_value,left,right,limits,images,rows,inner,channels,workers,
        sizes,Unit{},bias,bias_values,left_model,right_model,left_cover,right_cover)

def bias_from(+geometry: Cells.Cells,+outputs: U32,+content: Buffer.Content<Tile.Tile8>,
  +read: Gemm.Both<{Buffer.content_length(~Tile.Tile8,content) == Shape.elements([Gemm.blocks(outputs)]) : Nat},
    Contract.Values(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,geometry},content)>) ->
  Lanes.Bias<content,outputs>:
  Gemm.Both{+length,+values} = read
  zero_bias(geometry,outputs,content,%SharedContract.single_bias_size(outputs) : {Buffer.content_length(~Tile.Tile8,content) == _ : Nat} length,values)

def panel_modeled(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +logical_room: {Nat.is_le(Shape.elements([count,rows,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(Shape.elements([count,inner,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lm: Model.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,
  +left_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry((count * rows : U32),inner)},Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[count,rows,Gemm.blocks(inner)],lf,Tensors.Usable{},lw}),
  +wm: Model.Model<Tile.Tile8>,+wf: Tile.Tile8,+ww: U32,
  +right_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((count * inner : U32),outputs)},Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[count,inner,Gemm.blocks(outputs)],wf,Tensors.Usable{},ww}),
  +bm: Model.Model<Tile.Tile8>,+bf: Tile.Tile8,+bw: U32,
  +bias_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),
    Representation.Lifted{0.0,Samples.row_geometry(1,outputs)},Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(outputs)],bf,Tensors.Usable{},bw})) ->
  Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],
    Matrix.scheduled_run(Matrix.ImagePanels{},count,rows,inner,outputs,workers,
      Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[count,rows,Gemm.blocks(inner)],lf,Tensors.Usable{},lw},
      Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[count,inner,Gemm.blocks(outputs)],wf,Tensors.Usable{},ww},
      Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(outputs)],bf,Tensors.Usable{},bw})):
  +left_shape = {[count,rows,Gemm.blocks(inner)] : List<&2,U32>}
  +weight_shape = {[count,inner,Gemm.blocks(outputs)] : List<&2,U32>}
  +bias_shape = {[Gemm.blocks(outputs)] : List<&2,U32>}
  +left_reader = {Representation.Lifted{left,Samples.row_geometry((count * rows : U32),inner)} : Representation.Lifted<Left>}
  +right_reader = {Representation.Lifted{right,Samples.row_geometry((count * inner : U32),outputs)} : Representation.Lifted<Right>}
  +bias_reader = {Representation.Lifted{0.0,Samples.row_geometry(1,outputs)} : Representation.Lifted<F32>}
  +input_room = Count.counted_room(limits,left_shape,
    Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),left_reader,Model.pack(~Tile.Tile8,lm),left_shape,lf,lw,left_contract),
    default_capacity)
  +weights_room = Count.counted_room(limits,weight_shape,
    Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),right_reader,Model.pack(~Tile.Tile8,wm),weight_shape,wf,ww,right_contract),
    default_capacity)
  +logical_positive = three_positive(count,rows,outputs,count_positive,rows_positive,outputs_positive)
  +sizes = {PanelSizes{default_capacity,count_positive,head_room(limits,count,rows,outputs,rows_positive,outputs_positive,logical_room),rows_positive,
      middle_room(limits,count,rows,outputs,count_positive,outputs_positive,logical_room),inner_positive,inner_room,outputs_positive,outputs_room,
      input_room,weights_room,CellRoom.cells_room(limits,[count,rows,outputs],2n,Grid.Innermost{},logical_positive,logical_room),logical_room,right_room} :
    PanelSizes<limits,count,rows,inner,outputs>}
  +offset = SharedContract.joined_buffer(bm,bias_shape,bf)
  +bias_values = bias_from(Samples.row_geometry(1,outputs),outputs,Buffer.content(~Tile.Tile8,offset),
    SharedContract.operand_read(~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),bias_reader,bm,bias_shape,bf,bw,bias_contract))
  +left_cover = Windows.shaped_covered(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),left_reader,left_shape,Model.pack(~Tile.Tile8,lm),lf,
    Tensors.Usable{},lw,Gemm.Both{Global.Usable{lm,lf,lw,{==}},left_contract})
  +right_cover = Windows.shaped_covered(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),right_reader,weight_shape,Model.pack(~Tile.Tile8,wm),wf,
    Tensors.Usable{},ww,Gemm.Both{Global.Usable{wm,wf,ww,{==}},right_contract})
  %Buffer.remodeled(~Tile.Tile8,Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(bias_shape),bf,Model.pack(~Tile.Tile8,bm))) :
    PanelResult(~Left,~left_value,~Right,~right_value,left,right,count,rows,inner,outputs,workers,
      Matrix.panel_products(U32.is_zero((inner % 16 : U32)),count,rows,inner,outputs,workers,Model.pack(~Tile.Tile8,lm),Model.pack(~Tile.Tile8,wm),_))
  panel_kernel(~Left,~left_value,~Right,~right_value,U32.is_zero((inner % 16 : U32)),left,right,limits,count,rows,inner,outputs,workers,sizes,{==},
    offset,bias_values,lm,wm,left_cover,right_cover)

def panel_shaped(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +logical_room: {Nat.is_le(Shape.elements([count,rows,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(Shape.elements([count,inner,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -input: Tensors.Tensor<Tile.Tile8>,-weights: Tensors.Tensor<Tile.Tile8>,-offset: Tensors.Tensor<Tile.Tile8>,
  input_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Samples.row_geometry((count * rows : U32),inner)},[count,rows,Gemm.blocks(inner)],input),
  weights_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
    Representation.Lifted{right,Samples.row_geometry((count * inner : U32),outputs)},[count,inner,Gemm.blocks(outputs)],weights),
  offset_proof: Global.Shaped(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),
    Representation.Lifted{0.0,Samples.row_geometry(1,outputs)},[Gemm.blocks(outputs)],offset)) ->
  Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],
    Matrix.scheduled_run(Matrix.ImagePanels{},count,rows,inner,outputs,workers,input,weights,offset)):
  Gemm.Both{Global.Usable{+lm,+lf,+lw,+input_eq},+read_input} = input_proof
  Gemm.Both{Global.Usable{+wm,+wf,+ww,+weight_eq},+read_weights} = weights_proof
  Gemm.Both{Global.Usable{+bm,+bf,+bw,+offset_eq},+read_offset} = offset_proof
  -input_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,lm),[count,rows,Gemm.blocks(inner)],lf,Tensors.Usable{},lw} : Tensors.Tensor<Tile.Tile8>}
  -weight_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,wm),[count,inner,Gemm.blocks(outputs)],wf,Tensors.Usable{},ww} : Tensors.Tensor<Tile.Tile8>}
  -bias_cells = {Tensors.Tensor{Model.pack(~Tile.Tile8,bm),[Gemm.blocks(outputs)],bf,Tensors.Usable{},bw} : Tensors.Tensor<Tile.Tile8>}
  +logical = {Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)} : Logical.Batched<Left,Right>}
  +shape = {[count,rows,outputs] : List<&2,U32>}
  %input_eq : Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),logical,shape,
    Matrix.scheduled_run(Matrix.ImagePanels{},count,rows,inner,outputs,workers,_,weights,offset))
  %weight_eq : Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),logical,shape,
    Matrix.scheduled_run(Matrix.ImagePanels{},count,rows,inner,outputs,workers,input_cells,_,offset))
  %offset_eq : Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),logical,shape,
    Matrix.scheduled_run(Matrix.ImagePanels{},count,rows,inner,outputs,workers,input_cells,weight_cells,_))
  panel_modeled(~Left,~left_value,~Right,~right_value,left,right,limits,count,rows,inner,outputs,workers,default_capacity,count_positive,rows_positive,inner_positive,
    outputs_positive,inner_room,outputs_room,logical_room,right_room,lm,lf,lw,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,input_cells,input,input_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry((count * rows : U32),inner)},_)
    read_input,wm,wf,ww,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,weight_cells,weights,weight_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Samples.row_geometry((count * inner : U32),outputs)},_)
    read_weights,bm,bf,bw,
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,bias_cells,offset,offset_eq) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,Samples.row_geometry(1,outputs)},_)
    read_offset)

# Rows of left [count,rows,inner] and a right [count,inner,outputs] of every
# image, both with their lanes innermost, and a bias of +0 [outputs] hold the
# products of the image panels.
def panel_arranged(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +left: Left,+right: Right,+limits: Machine.Limits,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+outputs_positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool},
  +inner_room: {Nat.is_le(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +logical_room: {Nat.is_le(Shape.elements([count,rows,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(Shape.elements([count,inner,outputs]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  input: Tiled.Tiled,+input_proof: LaneLayout.Arranged(~Left,~left_value,left,[count,rows,inner],input,2n,Grid.Innermost{}),
  weights: Tiled.Tiled,+weights_proof: LaneLayout.Arranged(~Right,~right_value,right,[count,inner,outputs],weights,2n,Grid.Innermost{}),
  offset: Tiled.Tiled,+offset_proof: LaneLayout.Arranged(~F32,~FullValues.constant(~F32),0.0,[outputs],offset,0n,Grid.Innermost{})) ->
  Public.Contract(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(outputs)},[count,rows,outputs],
    Matrix.scheduled_lines(Matrix.ImagePanels{},count,rows,inner,outputs,workers,Matrix.innermost(input,2n),weights,Matrix.innermost(offset,0n))):
  Gemm.Both{input_form,+input_tiles} = input_proof
  Gemm.Both{weight_form,+weight_tiles} = weights_proof
  Gemm.Both{offset_form,+offset_tiles} = offset_proof
  +left_shape = {[count,rows,inner] : List<&2,U32>}
  +right_shape = {[count,inner,outputs] : List<&2,U32>}
  +bias_shape = {[outputs] : List<&2,U32>}
  -input_cells = Matrix.innermost(input,2n)
  -weight_cells = Matrix.innermost(weights,2n)
  -bias_cells = Matrix.innermost(offset,0n)
  +input_ready = {%Samples.matrix_geometry_of(count,rows,inner,Grid.Innermost{}) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,_},[count,rows,Gemm.blocks(inner)],input_cells)
    Public.cells_shaped(~Left,~left_value,left,left_shape,2n,Grid.lanes_at(left_shape,2n),Grid.Innermost{},Matrix.innermost(input,2n),input_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Samples.row_geometry((count * rows : U32),inner)},
      [count,rows,Gemm.blocks(inner)],input_cells)}
  +weight_ready = {%Samples.matrix_geometry_of(count,inner,outputs,Grid.Innermost{}) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,_},[count,inner,Gemm.blocks(outputs)],weight_cells)
    Public.cells_shaped(~Right,~right_value,right,right_shape,2n,Grid.lanes_at(right_shape,2n),Grid.Innermost{},Matrix.innermost(weights,2n),weight_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Samples.row_geometry((count * inner : U32),outputs)},
      [count,inner,Gemm.blocks(outputs)],weight_cells)}
  +bias_ready = {%Samples.bias_geometry_of(outputs) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,_},[Gemm.blocks(outputs)],bias_cells)
    Public.cells_shaped(~F32,~FullValues.constant(~F32),0.0,bias_shape,0n,Grid.lanes_at(bias_shape,0n),Grid.Innermost{},Matrix.innermost(offset,0n),offset_tiles) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<F32>,~Representation.lifted(~F32,~FullValues.constant(~F32)),Representation.Lifted{0.0,Samples.row_geometry(1,outputs)},
      [Gemm.blocks(outputs)],bias_cells)}
  panel_shaped(~Left,~left_value,~Right,~right_value,left,right,limits,count,rows,inner,outputs,workers,default_capacity,count_positive,rows_positive,inner_positive,
    outputs_positive,inner_room,outputs_room,logical_room,right_room,input_cells,weight_cells,bias_cells,input_ready,weight_ready,bias_ready)
