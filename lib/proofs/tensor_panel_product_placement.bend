# Image panels place the cells of the rectangles observed by their tasks.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../traversal_partition.bend as Partition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_gemm_output_groups.bend as Groups
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_move_readback.bend as Held
import ./tensor_panel_product_tasks.bend as Panels
import ./tensor_pieces_model.bend as Model
import ./tensor_point_product_fragments.bend as Fragments
import ./tensor_point_product_placement.bend as Placed
import ./tensor_point_product_tasks.bend as Shared
import ./tensor_point_products.bend as Point
import ./tensor_shared_product_placement.bend as SharedPlacement
import ./tensor_task_runs.bend as Runs

def panel_cells(+work: Products.Shape) -> Nat:
  Products.Shape{+rows,+inner,+channels,+grid} = work
  Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))

type Layout<-limits: Machine.Limits,-work: Products.Shape,-images: U32> is Data:
  Layout{grid_facts: GridLaws.GridFacts<limits,Panels.panel_grid(work)>,
    grid_exact: {GridLaws.grid_cells(Panels.panel_grid(work)) == Nat.mul(U32.to_nat(images),U32.to_nat(Panels.image_width(work))) : Nat},
    width_exact: {U32.to_nat(Panels.image_width(work)) == panel_cells(work) : Nat},
    width_positive: {Nat.is_lt(0n,U32.to_nat(Panels.image_width(work))) == True{} : Bool},
    room: {Nat.is_le(GridLaws.grid_cells(Panels.panel_grid(work)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    run: Runs.Run(Pieces.ranges_of_grid(Panels.panel_grid(work)),1n,GridLaws.grid_cells(Panels.panel_grid(work)))}

type Dimensions<-limits: Machine.Limits,-rows: U32,-channels: U32> is Data:
  Dimensions{rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    channels_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}}

def panel_end(+first: Nat,+count: U32,+width: Nat,+one: {count == 1 : U32}) ->
  {Nat.add(Nat.mul(first,width),width) == Nat.mul(Nat.add(first,U32.to_nat(count)),width) : Nat}:
  %Equal.sym(U32,count,1,one) : {Nat.add(Nat.mul(first,width),width) == Nat.mul(Nat.add(first,U32.to_nat(_)),width) : Nat}
  +expanded = Index.distrib(first,1n,width)
  Equal.sym(Nat,Nat.mul(Nat.add(first,1n),width),Nat.add(Nat.mul(first,width),width),
    Equal.trans(Nat,Nat.mul(Nat.add(first,1n),width),Nat.add(Nat.mul(first,width),Nat.add(width,0n)),Nat.add(Nat.mul(first,width),width),expanded,
      Equal.cong(Nat,Nat,term => Nat.add(Nat.mul(first,width),term),Nat.add(width,0n),width,Addition.zero_right(width))))

type LeafLayout<-limits: Machine.Limits,-work: Products.Shape,-image: U32,-count: U32> is Data:
  LeafLayout{start: {U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,image))) == Nat.mul(U32.to_nat(image),U32.to_nat(Panels.image_width(work))) : Nat},
    width: {U32.to_nat(Panels.image_width(work)) == Fragments.rectangle_cells(Point.panel_geometry(work,image)) : Nat},
    end: {Nat.add(U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,image))),U32.to_nat(Panels.image_width(work))) ==
      Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(Panels.image_width(work))) : Nat},
    room: Fragments.RectangleReadRoom<limits,Point.panel_geometry(work,image)>,
    laid: Shared.Laid(Pieces.ranges_of_grid(Panels.panel_grid(work)),Point.clipped_rectangle(Point.panel_geometry(work,image)),
      U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,image))),
      Nat.add(U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,image))),Fragments.rectangle_cells(Point.panel_geometry(work,image))))}

def leaf_layout(+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +dims: Dimensions<limits,rows,channels>,+layout: Layout<limits,Products.Shape{rows,inner,channels,grid},images>,
  +image: U32,+count: U32,+one: {count == 1 : U32},+inside: {Nat.is_le(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(images)) == True{} : Bool}) ->
  LeafLayout<limits,Products.Shape{rows,inner,channels,grid},image,count>:
  Dimensions{+rows_room,+rows_pos,+channels_room,+channels_pos} = dims
  Layout{+grid_facts,+grid_exact,+width_exact,+width_positive,+room,+out_run} = layout
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.panel_geometry(work,image)
  +group_word = Gemm.blocks(channels)
  +width = Panels.image_width(work)
  +size = U32.to_nat(width)
  +out_first = (image * rows * group_word : U32)
  +low = Nat.mul(U32.to_nat(image),size)
  +high = Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),size)
  +cap = Machine.capacity(limits)
  +total = GridLaws.grid_cells(grid)
  +below = {%Equal.sym(Nat,total,Nat.mul(U32.to_nat(images),size),grid_exact) : {Nat.is_le(high,_) == True{} : Bool}
    Index.mul_le(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(images),size,inside) : {Nat.is_le(high,total) == True{} : Bool}}
  +high_room = Order.transitive(high,total,U32.to_nat(cap),below,room)
  +low_room = Order.transitive(low,high,U32.to_nat(cap),
    Index.mul_le(U32.to_nat(image),Nat.add(U32.to_nat(image),U32.to_nat(count)),size,Order.le_add(U32.to_nat(image),U32.to_nat(count))),high_room)
  +first_exact = Equal.trans(Nat,U32.to_nat(out_first),U32.to_nat((image * width : U32)),low,
    Equal.cong(U32,Nat,U32.to_nat,out_first,(image * width : U32),Arithmetic.multiply_associative(image,rows,group_word)),
    Arithmetic.product_exact(cap,Limits.capacity_fits(limits),image,width,U32.to_nat(image),size,{==},{==},low_room))
  +last_exact = Equal.trans(Nat,Nat.add(U32.to_nat(out_first),size),Nat.add(low,size),high,
    Equal.cong(Nat,Nat,start => Nat.add(start,size),U32.to_nat(out_first),low,first_exact),panel_end(U32.to_nat(image),count,size,one))
  +group_room = Groups.groups_fit(limits,channels,channels_room)
  +space = {Fragments.RectangleReadRoom{rows_pos,rows_room,Groups.groups_positive(limits,channels,channels_room,channels_pos),group_room,Groups.pairs_fewer(limits,channels,group_room),
    %width_exact : {Nat.is_le(Nat.add(U32.to_nat(out_first),_),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(U32.to_nat(out_first),size),high,last_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} high_room} :
    Fragments.RectangleReadRoom<limits,rectangle>}
  +native_laid = Panels.image_leaf_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,out_run,image,count,one,inside)
  LeafLayout{first_exact,width_exact,last_exact,space,
    %Equal.sym(Nat,U32.to_nat(out_first),low,first_exact) :
      Shared.Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(rectangle),_,Nat.add(U32.to_nat(out_first),panel_cells(work)))
    %width_exact :
      Shared.Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(rectangle),low,Nat.add(U32.to_nat(out_first),_))
    %Equal.sym(Nat,Nat.add(U32.to_nat(out_first),size),high,last_exact) :
      Shared.Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(rectangle),low,_)
    native_laid}

def leaf_plan(~run: Point.Run(),+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+image: U32,+count: U32,
  +operands: Panels.Operands,+result: Shared.TaskModel(),+origin: Buffer.Content<Tile.Tile8>,
  layout: LeafLayout<limits,Products.Shape{rows,inner,channels,grid},image,count>,emitted: Panels.Emitted<run,Products.Shape{rows,inner,channels,grid},image,operands,result,origin>) ->
  Placed.Plan<limits,Pieces.ranges_of_grid(grid),Nat.mul(U32.to_nat(image),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid}))),
    Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid})))>:
  Panels.Operands{+left,+right} = operands
  LeafLayout{+first_exact,+width_exact,+last_exact,+space,+laid} = layout
  Panels.Emitted{+bias,+contents,same_bias} = emitted
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.panel_geometry(work,image)
  +stage = Point.panel_stage_model(~run,work,image,Held.pack(~Tile.Tile8,left),Held.pack(~Tile.Tile8,right),bias)
  Placed.RectanglePlan{rectangle,stage,result,Fragments.rectangle_start(rectangle),Panels.image_width(work),{==},width_exact,first_exact,last_exact,space,
    Equal.trans(Model.Content<Tile.Tile8>,Model.content(~Tile.Tile8,Placed.task_pieces(result)),Point.task_contents(Shared.pack_task(result)),Point.rectangle_model(rectangle,stage),
      Equal.sym(Model.Content<Tile.Tile8>,Point.task_contents(Shared.pack_task(result)),Model.content(~Tile.Tile8,Placed.task_pieces(result)),Placed.task_contents(result)),contents),laid}

def observed_leaf(~run: Point.Run(),+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+image: U32,+count: U32,
  +operands: Panels.Operands,+result: Shared.TaskModel(),+origin: Buffer.Content<Tile.Tile8>,
  facts: LeafLayout<limits,Products.Shape{rows,inner,channels,grid},image,count>,emitted: Panels.Emitted<run,Products.Shape{rows,inner,channels,grid},image,operands,result,origin>) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),Nat.mul(U32.to_nat(image),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid}))),
    Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid}))),result>:
  Panels.Operands{+left,+right} = operands
  LeafLayout{+first_exact,+width_exact,+last_exact,+space,+laid} = facts
  Panels.Emitted{+bias,+contents,+same_bias} = emitted
  Placed.ObservedPlan{leaf_plan(~run,limits,rows,inner,channels,grid,image,count,operands,result,origin,facts,emitted),{==}}

def seal_leaf(~run: Point.Run(),+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,
  +dims: Dimensions<limits,rows,channels>,+layout: Layout<limits,Products.Shape{rows,inner,channels,grid},images>,
  +image: U32,+count: U32,+operands: Panels.Operands,+result: Shared.TaskModel(),+origin: Buffer.Content<Tile.Tile8>,+one: {count == 1 : U32},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(images)) == True{} : Bool},
  +emitted: Panels.Emitted<run,Products.Shape{rows,inner,channels,grid},image,operands,result,origin>) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),Nat.mul(U32.to_nat(image),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid}))),
    Nat.mul(Nat.add(U32.to_nat(image),U32.to_nat(count)),U32.to_nat(Panels.image_width(Products.Shape{rows,inner,channels,grid}))),result>:
  observed_leaf(~run,limits,rows,inner,channels,grid,image,count,operands,result,origin,leaf_layout(limits,images,rows,inner,channels,grid,dims,layout,image,count,one,inside),emitted)

def coordinate(+work: Products.Shape,+image: Nat) -> Nat:
  Nat.mul(image,U32.to_nat(Panels.image_width(work)))

def join_panels(+limits: Machine.Limits,+work: Products.Shape,+left: Pieces.Ranges,+right: Pieces.Ranges,
  +left_result: Shared.TaskModel(),+right_result: Shared.TaskModel(),
  +glue: {Runs.range_high(left) == Runs.range_low(right) : Nat},
  +left_order: {Nat.is_le(Runs.range_low(left),Runs.range_high(left)) == True{} : Bool},
  +right_order: {Nat.is_le(Runs.range_low(right),Runs.range_high(right)) == True{} : Bool},
  +left_seen: Placed.ObservedPlan<limits,Pieces.ranges_of_grid(Panels.panel_grid(work)),coordinate(work,Runs.range_low(left)),coordinate(work,Runs.range_high(left)),left_result>,
  +right_seen: Placed.ObservedPlan<limits,Pieces.ranges_of_grid(Panels.panel_grid(work)),coordinate(work,Runs.range_low(right)),coordinate(work,Runs.range_high(right)),right_result>) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(Panels.panel_grid(work)),coordinate(work,Runs.range_low(left)),coordinate(work,Runs.range_high(right)),Partition.Fork{left_result,right_result}>:
  +layout = Pieces.ranges_of_grid(Panels.panel_grid(work))
  +width = U32.to_nat(Panels.image_width(work))
  +low = coordinate(work,Runs.range_low(left))
  +middle = coordinate(work,Runs.range_high(left))
  +second = coordinate(work,Runs.range_low(right))
  +high = coordinate(work,Runs.range_high(right))
  +edge = Equal.sym(Nat,middle,second,Equal.cong(Nat,Nat,number => coordinate(work,number),Runs.range_high(left),Runs.range_low(right),glue))
  Placed.join_plans(limits,layout,low,middle,high,left_result,right_result,Index.mul_le(Runs.range_low(left),Runs.range_high(left),width,left_order),
    %edge : {Nat.is_le(_,high) == True{} : Bool} Index.mul_le(Runs.range_low(right),Runs.range_high(right),width,right_order),
    left_seen,%edge : Placed.ObservedPlan<limits,layout,_,high,right_result> right_seen)

def tree_plan(~run: Point.Run(),+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+images: U32,
  +dims: Dimensions<limits,rows,channels>,+layout: Layout<limits,Products.Shape{rows,inner,channels,grid},images>,
  tree: Panels.OperandTree(),output: Shared.TaskModel(),+origin: Buffer.Content<Tile.Tile8>,
  +runs: Runs.Run(Panels.operand_ranges(tree),1n,U32.to_nat(images)),+ones: Panels.Singletons(Panels.operand_ranges(tree)),
  evidence: Panels.Emissions(~run,Products.Shape{rows,inner,channels,grid},tree,output,origin)) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),
    coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_low(Panels.operand_ranges(tree))),
    coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_high(Panels.operand_ranges(tree))),output>:
  match tree output:
    case Partition.Leaf{+operands,+first,+count,+details} +result:
      seal_leaf(~run,limits,images,rows,inner,channels,grid,dims,layout,first,count,operands,result,origin,ones,
        Runs.fits_inside(1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(images),runs),evidence)
    case Partition.Fork{+left,+right} Partition.Fork{+left_output,+right_output}:
      Held.Both{+left_run,+rest} = runs
      Held.Both{+right_run,+glue} = rest
      Held.Both{+left_ones,+right_ones} = ones
      Held.Both{+left_emitted,+right_emitted} = evidence
      +left_ranges = Panels.operand_ranges(left)
      +right_ranges = Panels.operand_ranges(right)
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      join_panels(limits,work,left_ranges,right_ranges,left_output,right_output,glue,
        SharedPlacement.run_order(left_ranges,1n,U32.to_nat(images),Runs.run_fits(left_ranges,1n,U32.to_nat(images),left_run)),
        SharedPlacement.run_order(right_ranges,1n,U32.to_nat(images),Runs.run_fits(right_ranges,1n,U32.to_nat(images),right_run)),
        tree_plan(~run,limits,rows,inner,channels,grid,images,dims,layout,left,left_output,origin,left_run,left_ones,left_emitted),
        tree_plan(~run,limits,rows,inner,channels,grid,images,dims,layout,right,right_output,origin,right_run,right_ones,right_emitted))
    case Partition.Fork{+left,+right} Partition.Leaf{+pieces,+first,+count,+detail}:
      Empty.absurd(Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),
        coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_low(Panels.operand_ranges(left))),
        coordinate(Products.Shape{rows,inner,channels,grid},Runs.range_high(Panels.operand_ranges(right))),Partition.Leaf{pieces,first,count,detail}>,evidence)
