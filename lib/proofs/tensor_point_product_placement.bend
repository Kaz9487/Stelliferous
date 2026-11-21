# Coherent readers for the placed fragments of a point-product task tree.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../default_machine_limits.bend as Default
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../traversal_partition.bend as Partition
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./tensor_global_shape_count.bend as Global
import ./tensor_grid_ranges.bend as Grids
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_model.bend as Model
import ./tensor_placed.bend as Placed
import ./tensor_placed_parts.bend as Parts
import ./tensor_point_product_fragments.bend as Fragments
import ./tensor_point_product_tasks.bend as Tasks
import ./tensor_point_products.bend as Point
import ./tensor_shape_refinement.bend as Shape
import ./tensor_storage_count.bend as Count
import ./tensor_task_runs.bend as Runs

type Reader is Data:
  ReadRectangle{source: Fragments.RectangleRead}
  ReadFork{middle: Nat,left: Reader,right: Reader}

def read(reader: Reader,index: U32) -> Tile.Tile8:
  match reader:
    case ReadRectangle{source}: Fragments.cell_value(source,index)
    case ReadFork{middle,left,right}:
      +index = index
      Bool.pick(Tile.Tile8,Nat.is_lt(U32.to_nat(index),middle),read(left,index),read(right,index))

def InTree(source: Fragments.RectangleRead,reader: Reader) -> Data:
  match reader:
    case ReadRectangle{+rectangle}: {source == rectangle : Fragments.RectangleRead}
    case ReadFork{+middle,+left,+right}:
      Placed.Side<Gemm.Both<{Nat.is_le(Fragments.upper_of(source),middle) == True{} : Bool},InTree(source,left)>,
        Gemm.Both<{Nat.is_le(middle,Fragments.lower_of(source)) == True{} : Bool},InTree(source,right)>>

def member_read(+source: Fragments.RectangleRead,+reader: Reader,+member: InTree(source,reader),+index: U32,
  +begins: {Nat.is_le(Fragments.lower_of(source),U32.to_nat(index)) == True{} : Bool},
  +ends: {Nat.is_lt(U32.to_nat(index),Fragments.upper_of(source)) == True{} : Bool}) ->
  {Fragments.cell_value(source,index) == read(reader,index) : Tile.Tile8}:
  match reader:
    case ReadRectangle{+rectangle}:
      %member : {Fragments.cell_value(source,index) == Fragments.cell_value(_,index) : Tile.Tile8}
      {==}
    case ReadFork{+middle,+left,+right}:
      match member:
        case Placed.OnLeft{Gemm.Both{+before,+left_member}}:
          +inside = Order.lt_le(U32.to_nat(index),Fragments.upper_of(source),middle,ends,before)
          %Equal.sym(Bool,Nat.is_lt(U32.to_nat(index),middle),True{},inside) :
            {Fragments.cell_value(source,index) == Bool.pick(Tile.Tile8,_,read(left,index),read(right,index)) : Tile.Tile8}
          member_read(source,left,left_member,index,begins,ends)
        case Placed.OnRight{Gemm.Both{+after,+right_member}}:
          +outside = Order.not_below(middle,U32.to_nat(index),Order.transitive(middle,Fragments.lower_of(source),U32.to_nat(index),after,begins))
          %Equal.sym(Bool,Nat.is_lt(U32.to_nat(index),middle),False{},outside) :
            {Fragments.cell_value(source,index) == Bool.pick(Tile.Tile8,_,read(left,index),read(right,index)) : Tile.Tile8}
          member_read(source,right,right_member,index,begins,ends)

def content_holds(+reader: Reader,content: Model.Content<Tile.Tile8>,facts: Fragments.RestatedPieces(~Reader,~read,reader,content)) ->
  Model.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Reader,~read,reader),content):
  match content:
    case Model.PieceContent{+buffer,+start,+length}: facts
    case Model.Joined{+left,+right}:
      Gemm.Both{+left_holds,+right_holds} = facts
      Gemm.Both{content_holds(reader,left,left_holds),content_holds(reader,right,right_holds)}

def rectangle_parts(+limits: Machine.Limits,+layout: Pieces.Ranges,+rectangle: Products.Rectangle,+stage: Buffer.Content<Tile.Tile16>,
  +reader: Reader,+first: U32,+count: U32,+pieces: Model.Model<Tile.Tile8>,
  +space: Fragments.RectangleReadRoom<limits,rectangle>,
  +member: InTree(Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},reader),
  +contents: {Model.content(~Tile.Tile8,pieces) == Point.rectangle_model(rectangle,stage) : Model.Content<Tile.Tile8>},
  +start: {U32.to_nat(first) == U32.to_nat(Fragments.rectangle_start(rectangle)) : Nat},
  +length: {U32.to_nat(count) == Fragments.rectangle_cells(rectangle) : Nat},
  +laid: Tasks.Laid(layout,Point.clipped_rectangle(rectangle),U32.to_nat(Fragments.rectangle_start(rectangle)),
    Nat.add(U32.to_nat(Fragments.rectangle_start(rectangle)),Fragments.rectangle_cells(rectangle)))) ->
  Held.Both<Parts.PartsAt(~Tile.Tile8,~Reader,~read,reader,layout,U32.to_nat(first),pieces),
    {Parts.pieces_end(~Tile.Tile8,pieces,U32.to_nat(first)) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}>:
  +model = Point.rectangle_model(rectangle,stage)
  +region = Point.clipped_rectangle(rectangle)
  +found = Fragments.rectangle_restated(~Reader,~read,~InTree,~member_read,limits,rectangle,stage,reader,member,layout,space,laid)
  +facts = {%Equal.sym(Model.Content<Tile.Tile8>,Model.content(~Tile.Tile8,pieces),model,contents) :
    Model.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Reader,~read,reader),_)
    content_holds(reader,model,found) : Model.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Reader,~read,reader),Model.content(~Tile.Tile8,pieces))}
  +shape = Equal.trans(Pieces.Ranges,Model.layout(~Tile.Tile8,pieces),Model.content_layout(~Tile.Tile8,model),region,
    %contents : {Model.layout(~Tile.Tile8,pieces) == Model.content_layout(~Tile.Tile8,_) : Pieces.Ranges}
    Model.layout_matches_content(~Tile.Tile8,pieces),Point.rectangle_layout(rectangle,stage))
  %Equal.sym(Nat,U32.to_nat(first),U32.to_nat(Fragments.rectangle_start(rectangle)),start) :
    Held.Both<Parts.PartsAt(~Tile.Tile8,~Reader,~read,reader,layout,_,pieces),
      {Parts.pieces_end(~Tile.Tile8,pieces,_) == Nat.add(_,U32.to_nat(count)) : Nat}>
  %Equal.sym(Nat,U32.to_nat(count),Fragments.rectangle_cells(rectangle),length) :
    Held.Both<Parts.PartsAt(~Tile.Tile8,~Reader,~read,reader,layout,U32.to_nat(Fragments.rectangle_start(rectangle)),pieces),
      {Parts.pieces_end(~Tile.Tile8,pieces,U32.to_nat(Fragments.rectangle_start(rectangle))) ==
        Nat.add(U32.to_nat(Fragments.rectangle_start(rectangle)),_) : Nat}>
  Parts.laid_task(~Tile.Tile8,~Reader,~read,reader,layout,region,pieces,U32.to_nat(Fragments.rectangle_start(rectangle)),
    Nat.add(U32.to_nat(Fragments.rectangle_start(rectangle)),Fragments.rectangle_cells(rectangle)),shape,facts,laid)

def task_pieces(task: Tasks.TaskModel()) -> Model.Model<Tile.Tile8>:
  match task:
    case Partition.Leaf{pieces,first,count,detail}: pieces
    case Partition.Fork{left,right}: Model.Fork{task_pieces(left),task_pieces(right)}

def task_contents(+task: Tasks.TaskModel()) ->
  {Point.task_contents(Tasks.pack_task(task)) == Model.content(~Tile.Tile8,task_pieces(task)) : Model.Content<Tile.Tile8>}:
  match task:
    case Partition.Leaf{+pieces,first,count,detail}: Model.packed_contents(~Tile.Tile8,pieces)
    case Partition.Fork{+left,+right}:
      %task_contents(left) : {Model.Joined{Point.task_contents(Tasks.pack_task(left)),Point.task_contents(Tasks.pack_task(right))} ==
        Model.Joined{_,Model.content(~Tile.Tile8,task_pieces(right))} : Model.Content<Tile.Tile8>}
      %task_contents(right) : {Model.Joined{Point.task_contents(Tasks.pack_task(left)),Point.task_contents(Tasks.pack_task(right))} ==
        Model.Joined{Point.task_contents(Tasks.pack_task(left)),_} : Model.Content<Tile.Tile8>}
      {==}

def pieces_finish(task: Tasks.TaskModel(),+start: Nat) ->
  {Parts.pieces_end(~Tile.Tile8,task_pieces(task),start) == Parts.tasks_end(~Tile.Tile8,task,start) : Nat}:
  match task:
    case Partition.Leaf{pieces,first,count,detail}: {==}
    case Partition.Fork{+left,+right}:
      %Equal.sym(Nat,Parts.pieces_end(~Tile.Tile8,task_pieces(left),start),Parts.tasks_end(~Tile.Tile8,left,start),pieces_finish(left,start)) :
        {Parts.pieces_end(~Tile.Tile8,task_pieces(right),_) == Parts.tasks_end(~Tile.Tile8,right,Parts.tasks_end(~Tile.Tile8,left,start)) : Nat}
      pieces_finish(right,Parts.tasks_end(~Tile.Tile8,left,start))

def tasks_at(+reader: Reader,+layout: Pieces.Ranges,task: Tasks.TaskModel(),+start: Nat,
  parts: Parts.PartsAt(~Tile.Tile8,~Reader,~read,reader,layout,start,task_pieces(task))) ->
  Parts.TasksParts(~Tile.Tile8,~Reader,~read,reader,layout,start,task):
  match task:
    case Partition.Leaf{pieces,first,count,detail}: parts
    case Partition.Fork{+left,+right}:
      Gemm.Both{+first,+second} = parts
      Gemm.Both{tasks_at(reader,layout,left,start,first),
        %pieces_finish(left,start) : Parts.TasksParts(~Tile.Tile8,~Reader,~read,reader,layout,_,right)
        tasks_at(reader,layout,right,Parts.pieces_end(~Tile.Tile8,task_pieces(left),start),second)}

def parts_from_rectangle(+reader: Reader,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+task: Tasks.TaskModel(),
  +parts: Held.Both<Parts.PartsAt(~Tile.Tile8,~Reader,~read,reader,layout,low,task_pieces(task)),{Parts.pieces_end(~Tile.Tile8,task_pieces(task),low) == high : Nat}>) ->
  Held.Both<Parts.TasksParts(~Tile.Tile8,~Reader,~read,reader,layout,low,task),{Parts.tasks_end(~Tile.Tile8,task,low) == high : Nat}>:
  Held.Both{+placed,+finish} = parts
  Held.Both{tasks_at(reader,layout,task,low,placed),%pieces_finish(task,low) : {_ == high : Nat} finish}
type Plan<-limits: Machine.Limits,-layout: Pieces.Ranges,-low: Nat,-high: Nat> is Data:
  RectanglePlan{rectangle: Products.Rectangle,stage: Buffer.Content<Tile.Tile16>,tasks: Tasks.TaskModel(),
    first: U32,count: U32,
    start: {U32.to_nat(first) == U32.to_nat(Fragments.rectangle_start(rectangle)) : Nat},
    length: {U32.to_nat(count) == Fragments.rectangle_cells(rectangle) : Nat},
    low_exact: {U32.to_nat(first) == low : Nat},
    high_exact: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == high : Nat},
    space: Fragments.RectangleReadRoom<limits,rectangle>,
    contents: {Model.content(~Tile.Tile8,task_pieces(tasks)) == Point.rectangle_model(rectangle,stage) : Model.Content<Tile.Tile8>},
    laid: Tasks.Laid(layout,Point.clipped_rectangle(rectangle),U32.to_nat(Fragments.rectangle_start(rectangle)),
      Nat.add(U32.to_nat(Fragments.rectangle_start(rectangle)),Fragments.rectangle_cells(rectangle)))}
  SplitPlan{lower: Nat,upper: Nat,lower_exact: {lower == low : Nat},upper_exact: {upper == high : Nat},
    middle: Nat,low_middle: {Nat.is_le(low,middle) == True{} : Bool},middle_high: {Nat.is_le(middle,high) == True{} : Bool},
    left: Plan<limits,layout,low,middle>,right: Plan<limits,layout,middle,high>}

def planned_reader(-limits: Machine.Limits,-layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>) -> Reader:
  match plan:
    case RectanglePlan{rectangle,stage,tasks,first,count,start,length,low_exact,high_exact,space,contents,laid}:
      ReadRectangle{Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)}}
    case SplitPlan{lower,upper,lower_exact,upper_exact,+middle,low_middle,middle_high,left,right}:
      ReadFork{middle,planned_reader(limits,layout,low,middle,left),planned_reader(limits,layout,middle,high,right)}

def planned_tasks(-limits: Machine.Limits,-layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>) -> Tasks.TaskModel():
  match plan:
    case RectanglePlan{rectangle,stage,tasks,first,count,start,length,low_exact,high_exact,space,contents,laid}:
      tasks
    case SplitPlan{lower,upper,lower_exact,upper_exact,+middle,low_middle,middle_high,left,right}:
      Partition.Fork{planned_tasks(limits,layout,low,middle,left),planned_tasks(limits,layout,middle,high,right)}

type ObservedPlan<-limits: Machine.Limits,-layout: Pieces.Ranges,-low: Nat,-high: Nat,-task: Tasks.TaskModel()> is Data:
  ObservedPlan{plan: Plan<limits,layout,low,high>,exact: {planned_tasks(limits,layout,low,high,plan) == task : Tasks.TaskModel()}}

def join_plans(+limits: Machine.Limits,+layout: Pieces.Ranges,+low: Nat,+middle: Nat,+high: Nat,+left: Tasks.TaskModel(),+right: Tasks.TaskModel(),
  +first: {Nat.is_le(low,middle) == True{} : Bool},+second: {Nat.is_le(middle,high) == True{} : Bool},
  +lower: ObservedPlan<limits,layout,low,middle,left>,+upper: ObservedPlan<limits,layout,middle,high,right>) ->
  ObservedPlan<limits,layout,low,high,Partition.Fork{left,right}>:
  ObservedPlan{+left_plan,+left_exact} = lower
  ObservedPlan{+right_plan,+right_exact} = upper
  +before = planned_tasks(limits,layout,low,middle,left_plan)
  +after = planned_tasks(limits,layout,middle,high,right_plan)
  ObservedPlan{SplitPlan{low,high,{==},{==},middle,first,second,left_plan,right_plan},
    %left_exact : {Partition.Fork{before,after} == Partition.Fork{_,right} : Tasks.TaskModel()}
    %right_exact : {Partition.Fork{before,after} == Partition.Fork{before,_} : Tasks.TaskModel()}
    {==}}

def ReadersWithin(+limits: Machine.Limits,+layout: Pieces.Ranges,+low: Nat,+high: Nat,plan: Plan<limits,layout,low,high>,+reader: Reader) -> Data:
  match plan:
    case RectanglePlan{+rectangle,+stage,tasks,first,count,start,length,low_exact,high_exact,space,contents,laid}:
      InTree(Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},reader)
    case SplitPlan{lower,upper,lower_exact,upper_exact,+middle,low_middle,middle_high,+left,+right}:
      Gemm.Both<ReadersWithin(limits,layout,low,middle,left,reader),ReadersWithin(limits,layout,middle,high,right,reader)>

def readers_left(+limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>,
  +middle: Nat,+left_reader: Reader,+right_reader: Reader,+ordered: {Nat.is_le(high,middle) == True{} : Bool},
  within: ReadersWithin(limits,layout,low,high,plan,left_reader)) ->
  ReadersWithin(limits,layout,low,high,plan,ReadFork{middle,left_reader,right_reader}):
  match plan:
    case RectanglePlan{+rectangle,+stage,tasks,+first,+count,+start,+length,low_exact,+high_exact,space,contents,laid}:
      +edge = Nat.add(U32.to_nat(Fragments.rectangle_start(rectangle)),Fragments.rectangle_cells(rectangle))
      Placed.OnLeft{Gemm.Both{
        %start : {Nat.is_le(Nat.add(_,Fragments.rectangle_cells(rectangle)),middle) == True{} : Bool}
        %length : {Nat.is_le(Nat.add(U32.to_nat(first),_),middle) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,high_exact) : {Nat.is_le(_,middle) == True{} : Bool}
        ordered,within}}
    case SplitPlan{lower,+upper,lower_exact,+upper_exact,+between,low_middle,+middle_high,+left,+right}:
      Gemm.Both{+left_within,+right_within} = within
      Gemm.Both{readers_left(limits,layout,low,between,left,middle,left_reader,right_reader,Order.transitive(between,upper,middle,
          %Equal.sym(Nat,upper,high,upper_exact) : {Nat.is_le(between,_) == True{} : Bool} middle_high,
          %Equal.sym(Nat,upper,high,upper_exact) : {Nat.is_le(_,middle) == True{} : Bool} ordered),left_within),
        readers_left(limits,layout,between,high,right,middle,left_reader,right_reader,ordered,right_within)}

def readers_right(+limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>,
  +middle: Nat,+left_reader: Reader,+right_reader: Reader,+ordered: {Nat.is_le(middle,low) == True{} : Bool},
  within: ReadersWithin(limits,layout,low,high,plan,right_reader)) ->
  ReadersWithin(limits,layout,low,high,plan,ReadFork{middle,left_reader,right_reader}):
  match plan:
    case RectanglePlan{+rectangle,+stage,tasks,+first,count,+start,length,+low_exact,high_exact,space,contents,laid}:
      Placed.OnRight{Gemm.Both{
        %start : {Nat.is_le(middle,_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(first),low,low_exact) : {Nat.is_le(middle,_) == True{} : Bool}
        ordered,within}}
    case SplitPlan{+lower,upper,+lower_exact,upper_exact,+between,+low_middle,middle_high,+left,+right}:
      Gemm.Both{+left_within,+right_within} = within
      Gemm.Both{readers_right(limits,layout,low,between,left,middle,left_reader,right_reader,ordered,left_within),
        readers_right(limits,layout,between,high,right,middle,left_reader,right_reader,Order.transitive(middle,lower,between,
          %Equal.sym(Nat,lower,low,lower_exact) : {Nat.is_le(middle,_) == True{} : Bool} ordered,
          %Equal.sym(Nat,lower,low,lower_exact) : {Nat.is_le(_,between) == True{} : Bool} low_middle),right_within)}

def planned_reads(+limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>) ->
  ReadersWithin(limits,layout,low,high,plan,planned_reader(limits,layout,low,high,plan)):
  match plan:
    case RectanglePlan{rectangle,stage,tasks,first,count,start,length,low_exact,high_exact,space,contents,laid}: {==}
    case SplitPlan{lower,upper,lower_exact,upper_exact,+middle,low_middle,middle_high,+left,+right}:
      +before = planned_reader(limits,layout,low,middle,left)
      +after = planned_reader(limits,layout,middle,high,right)
      Gemm.Both{readers_left(limits,layout,low,middle,left,middle,before,after,Order.reflexive(middle),planned_reads(limits,layout,low,middle,left)),
        readers_right(limits,layout,middle,high,right,middle,before,after,Order.reflexive(middle),planned_reads(limits,layout,middle,high,right))}

def OrderedParts(+reader: Reader,+layout: Pieces.Ranges,+low: Nat,+high: Nat,+tasks: Tasks.TaskModel()) -> Data:
  Held.Both<Parts.TasksParts(~Tile.Tile8,~Reader,~read,reader,layout,low,tasks),{Parts.tasks_end(~Tile.Tile8,tasks,low) == high : Nat}>

def joined(+reader: Reader,+layout: Pieces.Ranges,-low: Nat,-middle: Nat,-high: Nat,+left: Tasks.TaskModel(),+right: Tasks.TaskModel(),
  +left_parts: OrderedParts(reader,layout,low,middle,left),+right_parts: OrderedParts(reader,layout,middle,high,right)) ->
  OrderedParts(reader,layout,low,high,Partition.Fork{left,right}):
  Held.Both{+before,+first_end} = left_parts
  Held.Both{+after,+second_end} = right_parts
  -finish = Parts.tasks_end(~Tile.Tile8,left,low)
  +moved = Equal.sym(Nat,finish,middle,first_end)
  Held.Both{Gemm.Both{before,%moved : Parts.TasksParts(~Tile.Tile8,~Reader,~read,reader,layout,_,right) after},
    %moved : {Parts.tasks_end(~Tile.Tile8,right,_) == high : Nat} second_end}

def planned_parts(+limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Plan<limits,layout,low,high>,
  +reader: Reader,within: ReadersWithin(limits,layout,low,high,plan,reader)) ->
  OrderedParts(reader,layout,low,high,planned_tasks(limits,layout,low,high,plan)):
  match plan:
    case RectanglePlan{+rectangle,+stage,+tasks,+first,+count,+start,+length,+low_exact,+high_exact,+space,+contents,+laid}:
      %low_exact : OrderedParts(reader,layout,_,high,tasks)
      %high_exact : OrderedParts(reader,layout,U32.to_nat(first),_,tasks)
      parts_from_rectangle(reader,layout,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),tasks,
        rectangle_parts(limits,layout,rectangle,stage,reader,first,count,task_pieces(tasks),space,within,contents,start,length,laid))
    case SplitPlan{lower,upper,lower_exact,upper_exact,+middle,+low_middle,+middle_high,+left,+right}:
      Gemm.Both{+left_within,+right_within} = within
      joined(reader,layout,low,middle,high,planned_tasks(limits,layout,low,middle,left),planned_tasks(limits,layout,middle,high,right),
        planned_parts(limits,layout,low,middle,left,reader,left_within),
        planned_parts(limits,layout,middle,high,right,reader,right_within))

def placed_reader(+limits: Machine.Limits,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tasks: Pieces.Ranges,+plan: Plan<limits,layout,Runs.range_low(tasks),Runs.range_high(tasks)>,
  +low: {Runs.range_low(tasks) == Runs.range_low(layout) : Nat},+high: {Runs.range_high(tasks) == Runs.range_high(layout) : Nat}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Reader,~read,planned_reader(limits,layout,Runs.range_low(tasks),Runs.range_high(tasks),plan)),layout,
    Tensors.placed(~Tile.Tile8,layout,Products.zero(),Tasks.pack_task(planned_tasks(limits,layout,Runs.range_low(tasks),Runs.range_high(tasks),plan)))>:
  +start = Runs.range_low(tasks)
  +end = Runs.range_high(tasks)
  +reader = planned_reader(limits,layout,start,end,plan)
  Parts.placed_run(~Tile.Tile8,~Reader,~read,reader,layout,Products.zero(),tasks,planned_tasks(limits,layout,start,end,plan),unit,bound,limits,run,room,low,high,
    planned_parts(limits,layout,start,end,plan,reader,planned_reads(limits,layout,start,end,plan)))

def shape_plan(+limits: Machine.Limits,+layout: Pieces.Ranges) -> Data:
  Plan<limits,layout,Runs.range_low(layout),Runs.range_high(layout)>

def installed(+limits: Machine.Limits,+layout: Pieces.Ranges,+plan: shape_plan(limits,layout)) -> Pieces.Pieces(~Tile.Tile8):
  Tensors.placed(~Tile.Tile8,layout,Products.zero(),Tasks.pack_task(planned_tasks(limits,layout,Runs.range_low(layout),Runs.range_high(layout),plan)))

def shape_contract(+limits: Machine.Limits,+shape: List<&2,U32>,+workers: U32,+unit: Nat,
  +room: {Nat.is_le(Shape.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +counted_layout: Grids.GridLayout<shape,Grid.piece_target(),unit>,
  +plan: shape_plan(limits,Pieces.ranges_of_grid(Tensors.grid_of(shape))),
  -actual: Pieces.Pieces(~Tile.Tile8),+same: {actual == installed(limits,Pieces.ranges_of_grid(Tensors.grid_of(shape)),plan) : Pieces.Pieces(~Tile.Tile8)}) ->
  Global.Contract(~Tile.Tile8,~Reader,~read,planned_reader(limits,Pieces.ranges_of_grid(Tensors.grid_of(shape)),
    Runs.range_low(Pieces.ranges_of_grid(Tensors.grid_of(shape))),Runs.range_high(Pieces.ranges_of_grid(Tensors.grid_of(shape))),plan),
    Tensors.Tensor{actual,shape,Products.zero(),Tensors.Usable{},workers}):
  Grids.GridLayout{+run,+low,+high} = counted_layout
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +begin = Runs.range_low(layout)
  +end = Runs.range_high(layout)
  +reader = planned_reader(limits,layout,begin,end,plan)
  +counted = Count.counted_within(limits,shape,room,default_capacity)
  -real = installed(limits,layout,plan)
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),actual,real,same) :
    Global.Contract(~Tile.Tile8,~Reader,~read,reader,Tensors.Tensor{_,shape,Products.zero(),Tensors.Usable{},workers})
  Placed.placed_contract(~Tile.Tile8,~Reader,~read,reader,shape,Products.zero(),workers,real,unit,counted,run,low,high,
    placed_reader(limits,layout,unit,Shape.elements(shape),run,room,layout,plan,{==},{==}))
