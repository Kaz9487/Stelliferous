# A point-product plan holds its logical values after clearing padded lanes.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../default_machine_limits.bend as Default
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_map.bend as TiledMap
import ../traversal_partition.bend as Partition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_tile_refinement.bend as TileLaws
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./storage_buffer_model.bend as Buffer
import ./tensor_global_shape_count.bend as Global
import ./tensor_grid_ranges.bend as Grids
import ./tensor_padded_pieces.bend as PiecePadding
import ./tensor_pieces_model.bend as Model
import ./tensor_point_product_fragments.bend as Fragments
import ./tensor_point_product_placement.bend as Placed
import ./tensor_point_product_tasks.bend as Tasks
import ./tensor_shape_refinement.bend as Shape
import ./tensor_task_runs.bend as Runs
import ./tensor_tiled_padded_cells.bend as Padded
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_restore_padding.bend as Restore

def cell_mask(~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,+padding: TiledMap.Padding,+reader: Reader,+word: U32) -> Tile.Tile8:
  Padding.cell(padding,U32.to_nat(word),read(reader,word))

def At(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+target: Target,+padding: TiledMap.Padding,+source: Fragments.RectangleRead,+position: Nat) -> Data:
  +index = U32.from_nat(Nat.add(Fragments.lower_of(source),position))
  {cell_mask(~Fragments.RectangleRead,~Fragments.cell_value,padding,source,index) == target_read(target,index) : Tile.Tile8}

def Masked(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,
  +target: Target,+padding: TiledMap.Padding,+limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Placed.Plan<limits,layout,low,high>) -> Data:
  match plan:
    case Placed.RectanglePlan{+rectangle,+stage,tasks,first,count,start,length,low_exact,high_exact,space,contents,laid}:
      Members.All(Nat,position => At(~Target,~target_read,target,padding,Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)},position),
        Ranges.range(Fragments.rectangle_cells(rectangle),0n))
    case Placed.SplitPlan{lower,upper,lower_exact,upper_exact,+middle,low_middle,middle_high,+left,+right}:
      Gemm.Both<Masked(~Target,~target_read,target,padding,limits,layout,low,middle,left),Masked(~Target,~target_read,target,padding,limits,layout,middle,high,right)>

def rectangle_cell(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+target: Target,+padding: TiledMap.Padding,
  +source: Fragments.RectangleRead,+length: Nat,+word: U32,
  +width: {Fragments.upper_of(source) == Nat.add(Fragments.lower_of(source),length) : Nat},
  +starts: {Nat.is_le(Fragments.lower_of(source),U32.to_nat(word)) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(word),Fragments.upper_of(source)) == True{} : Bool},
  +values: Members.All(Nat,position => At(~Target,~target_read,target,padding,source,position),Ranges.range(length,0n))) ->
  {cell_mask(~Fragments.RectangleRead,~Fragments.cell_value,padding,source,word) == target_read(target,word) : Tile.Tile8}:
  +first = Fragments.lower_of(source)
  +position = U32.to_nat(word)
  +offset = Nat.sub(position,first)
  +number = U32.from_nat(Nat.add(first,offset))
  +exact = Equal.trans(U32,number,U32.from_nat(position),word,
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(first,offset),position,Order.restore_shift(first,position,starts)),Arithmetic.native_roundtrip(word))
  +inside = Order.sub_lt(position,first,length,starts,%width : {Nat.is_lt(position,_) == True{} : Bool} before)
  %exact : {cell_mask(~Fragments.RectangleRead,~Fragments.cell_value,padding,source,_) == target_read(target,_) : Tile.Tile8}
  Members.all_member(Nat,position => At(~Target,~target_read,target,padding,source,position),offset,Ranges.range(length,0n),values,Members.range_member(length,0n,offset,inside))

def choose_cell(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,side: Bool,+target: Target,+padding: TiledMap.Padding,
  +middle: Nat,+left: Placed.Reader,+right: Placed.Reader,+word: U32,
  +same: {Nat.is_lt(U32.to_nat(word),middle) == side : Bool},
  before: @+smaller: {Nat.is_lt(U32.to_nat(word),middle) == True{} : Bool} ->
    {cell_mask(~Placed.Reader,~Placed.read,padding,left,word) == target_read(target,word) : Tile.Tile8},
  after: @+lower: {Nat.is_le(middle,U32.to_nat(word)) == True{} : Bool} ->
    {cell_mask(~Placed.Reader,~Placed.read,padding,right,word) == target_read(target,word) : Tile.Tile8}) ->
  {cell_mask(~Placed.Reader,~Placed.read,padding,Placed.ReadFork{middle,left,right},word) == target_read(target,word) : Tile.Tile8}:
  match side:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(U32.to_nat(word),middle),True{},same) :
        {Padding.cell(padding,U32.to_nat(word),Bool.pick(Tile.Tile8,_,Placed.read(left,word),Placed.read(right,word))) == target_read(target,word) : Tile.Tile8}
      before(same)
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(U32.to_nat(word),middle),False{},same) :
        {Padding.cell(padding,U32.to_nat(word),Bool.pick(Tile.Tile8,_,Placed.read(left,word),Placed.read(right,word))) == target_read(target,word) : Tile.Tile8}
      after(Order.not_less(U32.to_nat(word),middle,same))

def masked_cell(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+target: Target,+padding: TiledMap.Padding,
  +limits: Machine.Limits,+layout: Pieces.Ranges,-low: Nat,-high: Nat,plan: Placed.Plan<limits,layout,low,high>,
  values: Masked(~Target,~target_read,target,padding,limits,layout,low,high,plan),+word: U32,
  +starts: {Nat.is_le(low,U32.to_nat(word)) == True{} : Bool},
  +ends: {Nat.is_lt(U32.to_nat(word),high) == True{} : Bool}) ->
  {cell_mask(~Placed.Reader,~Placed.read,padding,Placed.planned_reader(limits,layout,low,high,plan),word) == target_read(target,word) : Tile.Tile8}:
  match plan:
    case Placed.RectanglePlan{+rectangle,+stage,+tasks,+first,+count,+start,+length,+low_exact,+high_exact,+space,+contents,+laid}:
      +source = {Fragments.RectangleRead{rectangle,Buffer.content_tree(~Tile.Tile16,stage)} : Fragments.RectangleRead}
      +begin = Fragments.lower_of(source)
      +num = Fragments.rectangle_cells(rectangle)
      +last = Fragments.upper_of(source)
      +edge = Nat.add(U32.to_nat(first),U32.to_nat(count))
      rectangle_cell(~Target,~target_read,target,padding,source,num,word,{==},
        %start : {Nat.is_le(_,U32.to_nat(word)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(first),low,low_exact) : {Nat.is_le(_,U32.to_nat(word)) == True{} : Bool} starts,
        %start : {Nat.is_lt(U32.to_nat(word),Nat.add(_,num)) == True{} : Bool}
        %length : {Nat.is_lt(U32.to_nat(word),Nat.add(U32.to_nat(first),_)) == True{} : Bool}
        %Equal.sym(Nat,edge,high,high_exact) : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} ends,
        values)
    case Placed.SplitPlan{lower,upper,lower_exact,upper_exact,+middle,+low_middle,+middle_high,+left,+right}:
      Gemm.Both{+left_values,+right_values} = values
      choose_cell(~Target,~target_read,Nat.is_lt(U32.to_nat(word),middle),target,padding,middle,
        Placed.planned_reader(limits,layout,low,middle,left),Placed.planned_reader(limits,layout,middle,high,right),word,{==},
        before => masked_cell(~Target,~target_read,target,padding,limits,layout,low,middle,left,left_values,word,starts,before),
        after => masked_cell(~Target,~target_read,target,padding,limits,layout,middle,high,right,right_values,word,after,ends))

type Checked<-Target: Data,-target_read: Target -> U32 -> Tile.Tile8,-target: Target,-padding: TiledMap.Padding,
  -limits: Machine.Limits,-layout: Pieces.Ranges,-low: Nat,-high: Nat,-tasks: Tasks.TaskModel()> is Data:
  Checked{plan: Placed.Plan<limits,layout,low,high>,exact: {Placed.planned_tasks(limits,layout,low,high,plan) == tasks : Tasks.TaskModel()},
    values: Masked(~Target,~target_read,target,padding,limits,layout,low,high,plan)}

def checked_join(~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+target: Target,+padding: TiledMap.Padding,
  +limits: Machine.Limits,+layout: Pieces.Ranges,+low: Nat,+middle: Nat,+high: Nat,+left: Tasks.TaskModel(),+right: Tasks.TaskModel(),
  +first: {Nat.is_le(low,middle) == True{} : Bool},+second: {Nat.is_le(middle,high) == True{} : Bool},
  +lower: Checked<Target,target_read,target,padding,limits,layout,low,middle,left>,
  +upper: Checked<Target,target_read,target,padding,limits,layout,middle,high,right>) ->
  Checked<Target,target_read,target,padding,limits,layout,low,high,Partition.Fork{left,right}>:
  Checked{+left_plan,+left_exact,+left_values} = lower
  Checked{+right_plan,+right_exact,+right_values} = upper
  +before = Placed.planned_tasks(limits,layout,low,middle,left_plan)
  +after = Placed.planned_tasks(limits,layout,middle,high,right_plan)
  Checked{Placed.SplitPlan{low,high,{==},{==},middle,first,second,left_plan,right_plan},
    %left_exact : {Partition.Fork{before,after} == Partition.Fork{_,right} : Tasks.TaskModel()}
    %right_exact : {Partition.Fork{before,after} == Partition.Fork{before,_} : Tasks.TaskModel()}
    {==},Gemm.Both{left_values,right_values}}

def MaskedLane(~Logical: Data,~read: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+tile: Tile.Tile8,+selected: Nat) -> Data:
  {Padded.kept(Representation.lane_count(geometry),group,selected,TileLaws.select_lane(selected,tile)) ==
    TileLaws.select_lane(selected,Representation.cell(~Logical,~read,logical,geometry,outer,group)) : F32}

def masked_lane(~Logical: Data,~read: Logical -> U32 -> F32,
  +logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+tile: Tile.Tile8,+selected: Nat,
  +small: {Nat.is_le(selected,7n) == True{} : Bool},
  agrees: @+active: {Nat.is_lt(Nat.add(selected,Nat.mul(group,8n)),Representation.lane_count(geometry)) == True{} : Bool} ->
    {TileLaws.select_lane(selected,tile) == read(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(selected,Nat.mul(group,8n))))) : F32}) ->
  MaskedLane(~Logical,~read,logical,geometry,outer,group,tile,selected):
  +lanes = Representation.lane_count(geometry)
  +index = Nat.add(selected,Nat.mul(group,8n))
  +scalar = read(logical,U32.from_nat(Representation.logical_index(geometry,outer,index)))
  +raw = TileLaws.select_lane(selected,tile)
  +target = Representation.cell(~Logical,~read,logical,geometry,outer,group)
  Equal.trans(F32,Padded.kept(lanes,group,selected,raw),Representation.lane(~Logical,~read,logical,geometry,outer,group,selected),TileLaws.select_lane(selected,target),
    Padded.kept_agree(lanes,group,selected,raw,scalar,Nat.is_lt(index,lanes),{==},agrees),
    Equal.sym(F32,TileLaws.select_lane(selected,target),Representation.lane(~Logical,~read,logical,geometry,outer,group,selected),
      Representation.cell_lane(~Logical,~read,logical,geometry,outer,group,selected,small)))

def ValidLanes(~Logical: Data,~read: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+tile: Tile.Tile8) -> Data:
  Members.All(Nat,selected => MaskedLane(~Logical,~read,logical,geometry,outer,group,tile,selected),Ranges.range(8n,0n))

def valid_tile(~Logical: Data,~read: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+tile: Tile.Tile8,
  lanes: ValidLanes(~Logical,~read,logical,geometry,outer,group,tile)) ->
  Padded.Valid<Representation.lane_count(geometry),group,tile,Representation.cell(~Logical,~read,logical,geometry,outer,group)>:
  Gemm.Both{a,Gemm.Both{b,Gemm.Both{c,Gemm.Both{d,Gemm.Both{e,Gemm.Both{f,Gemm.Both{g,Gemm.Both{h,stop}}}}}}}} = lanes
  Padded.Valid{a,b,c,d,e,f,g,h}

def decoded_valid(~Logical: Data,~read: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+word: U32,+tile: Tile.Tile8,
  +exact: {U32.to_nat(word) == Representation.cell_index(geometry,outer,group) : Nat},
  +outer_inside: {Nat.is_lt(outer,Representation.outer_count(geometry)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,Representation.group_count(geometry)) == True{} : Bool},
  +valid: Padded.Valid<Representation.lane_count(geometry),group,tile,Representation.cell(~Logical,~read,logical,geometry,outer,group)>) ->
  Padded.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),tile,
    Representation.lifted(~Logical,~read,Representation.Lifted{logical,geometry},word)>:
  Cells.Cells{+extent,+after,+lanes,+groups,+position} = geometry
  +shape = {Cells.Cells{extent,after,lanes,groups,position} : Cells.Cells}
  +index = Representation.cell_index(shape,outer,group)
  +number = U32.to_nat(word)
  +at = Representation.cell_group(shape,number)
  +held = Representation.lifted(~Logical,~read,Representation.Lifted{logical,shape},word)
  +target = Representation.cell(~Logical,~read,logical,shape,outer,group)
  +group_exact = {%Equal.sym(Nat,number,index,exact) : {Representation.cell_group(shape,_) == group : Nat}
    Representation.placed_group(position,U32.to_nat(extent),U32.to_nat(groups),outer,group,outer_inside,group_inside) :
    {at == group : Nat}}
  %Equal.sym(Nat,at,group,group_exact) : Padded.Valid<U32.to_nat(lanes),_,tile,held>
  %Equal.sym(Tile.Tile8,held,target,Representation.decoded(~Logical,~read,logical,shape,outer,group,word,exact,outer_inside,group_inside)) :
    Padded.Valid<U32.to_nat(lanes),group,tile,_>
  valid

def masked_decoded(~Logical: Data,~read: Logical -> U32 -> F32,+logical: Logical,+layout: Padded.Layout,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+word: U32,+tile: Tile.Tile8,
  +geometry_exact: {Padded.layout_geometry(layout) == geometry : Cells.Cells},
  +exact: {U32.to_nat(word) == Representation.cell_index(geometry,outer,group) : Nat},
  +inside: {Nat.is_lt(U32.to_nat(word),Padded.layout_total(layout)) == True{} : Bool},
  +outer_inside: {Nat.is_lt(outer,Representation.outer_count(geometry)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,Representation.group_count(geometry)) == True{} : Bool},
  +valid: Padded.Valid<Representation.lane_count(geometry),group,tile,Representation.cell(~Logical,~read,logical,geometry,outer,group)>) ->
  {Padding.cell(Padded.layout_padding(layout),U32.to_nat(word),tile) == Representation.lifted(~Logical,~read,Representation.Lifted{logical,geometry},word) : Tile.Tile8}:
  +number = U32.to_nat(word)
  +actual_geometry = Padded.layout_geometry(layout)
  +goal = Representation.lifted(~Logical,~read,Representation.Lifted{logical,geometry},word)
  PiecePadding.layout_cell(layout,word,inside,tile,goal,
    %Equal.sym(Cells.Cells,actual_geometry,geometry,geometry_exact) :
      Padded.Valid<Representation.lane_count(_),Representation.cell_group(_,number),tile,goal>
    decoded_valid(~Logical,~read,logical,geometry,outer,group,word,tile,exact,outer_inside,group_inside,valid))

def cleared_tiled(aligned: Bool,+padding: TiledMap.Padding,-cells: Tensors.Tensor<Tile.Tile8>,+axis: Nat,+lanes: U32,+position: Grid.Position) ->
  {Tiled.Tiled{Tiled.cells_of(TiledMap.clear_tail(aligned,padding,Tiled.Tiled{cells,axis,lanes,position})),axis,lanes,position} ==
    TiledMap.clear_tail(aligned,padding,Tiled.Tiled{cells,axis,lanes,position}) : Tiled.Tiled}:
  match aligned:
    case True{}: {==}
    case False{}: {==}

def cell_mask_value(~Reader: Data,~read: Reader -> U32 -> Tile.Tile8,+padding: TiledMap.Padding,+reader: Reader,+index: U32) ->
  {PiecePadding.padded_value(~Reader,~read,PiecePadding.Padded{padding,reader},index) == cell_mask(~Reader,~read,padding,reader,index) : Tile.Tile8}:
  %Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)) :
    {Padding.cell(padding,U32.to_nat(index),read(reader,_)) == cell_mask(~Reader,~read,padding,reader,index) : Tile.Tile8}
  {==}

def scope_mask(+scope: Restore.Scope<Placed.Reader>,+index: U32) ->
  {PiecePadding.padded_value(~Placed.Reader,~Placed.read,Restore.after_of(~Placed.Reader,scope),index) ==
    cell_mask(~Placed.Reader,~Placed.read,Restore.padding_of(~Placed.Reader,scope),Restore.before_of(~Placed.Reader,scope),index) : Tile.Tile8}:
  Restore.Scope{+reader,+padding,cap,cap_ok,fits} = scope
  cell_mask_value(~Placed.Reader,~Placed.read,padding,reader,index)

def sealed(~Logical: Data,~value: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+shape: List<&2,U32>,
  +axis: Nat,+lanes: U32,+position: Grid.Position,+workers: U32,+model: Model.Model<Tile.Tile8>,
  +reader: Placed.Reader,+scope: Restore.Scope<Placed.Reader>,
  +scope_reader: {Restore.before_of(~Placed.Reader,scope) == reader : Placed.Reader},
  +shape_exact: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +laid: Public.Laid<shape,axis,lanes,position,values>,
  +total_exact: {Restore.total_of(~Placed.Reader,scope) == Shape.elements(shape) : Nat},
  +word: U32,+room: {Nat.is_le(Shape.elements(shape),U32.to_nat(word)) == True{} : Bool},
  +raw: Global.Contract(~Tile.Tile8,~Placed.Reader,~Placed.read,reader,Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,Products.zero(),Tensors.Usable{},workers}),
  agrees: @+index: U32 -> @+inside: {Nat.is_lt(U32.to_nat(index),Shape.elements(shape)) == True{} : Bool} ->
    {cell_mask(~Placed.Reader,~Placed.read,Restore.padding_of(~Placed.Reader,scope),Restore.before_of(~Placed.Reader,scope),index) ==
      Representation.lifted(~Logical,~value,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},index) : Tile.Tile8}) ->
  Public.Contract(~Logical,~value,logical,values,TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,Products.zero(),Tensors.Usable{},workers},axis,lanes,position},
    Restore.padding_of(~Placed.Reader,scope)))):
  Restore.Scope{+stored,TiledMap.Padding{+width,+groups,+outer,+padded_position},+capacity,+cap_ok,+fits} = scope
  +padding = {TiledMap.Padding{width,groups,outer,padded_position} : TiledMap.Padding}
  +aligned = U32.is_zero((width % 8 : U32))
  +target = {Representation.Lifted{logical,Public.geometry_of(values,axis,position)} : Representation.Lifted<Logical>}
  -input = {Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,Products.zero(),Tensors.Usable{},workers} : Tensors.Tensor<Tile.Tile8>}
  -cleared = Tiled.cells_of(TiledMap.clear_tail(aligned,padding,Tiled.Tiled{input,axis,lanes,position}))
  +source = Restore.after_of(~Placed.Reader,scope)
  +cleared_contract = Restore.clear_shaped(~Placed.Reader,~Placed.read,aligned,scope,shape,Products.zero(),workers,axis,lanes,position,model,{==},total_exact,
    %Equal.sym(Placed.Reader,Restore.before_of(~Placed.Reader,scope),reader,scope_reader) :
      Global.Contract(~Tile.Tile8,~Placed.Reader,~Placed.read,_,input) raw)
  +restated = Global.restated_below(~Tile.Tile8,~PiecePadding.Padded<Placed.Reader>,~Representation.Lifted<Logical>,
    ~PiecePadding.padded_value(~Placed.Reader,~Placed.read),~Representation.lifted(~Logical,~value),~Shape.elements(shape),source,target,
    index => inside => Equal.trans(Tile.Tile8,PiecePadding.padded_value(~Placed.Reader,~Placed.read,source,index),
      cell_mask(~Placed.Reader,~Placed.read,padding,Restore.before_of(~Placed.Reader,scope),index),Representation.lifted(~Logical,~value,target,index),
      scope_mask(scope,index),agrees(index,inside)),
    shape,cleared,cleared_contract,Order.reflexive(Shape.elements(shape)),word,room)
  %cleared_tiled(aligned,padding,input,axis,lanes,position) : Public.Contract(~Logical,~value,logical,values,_)
  Public.laid_contract(~Logical,~value,logical,values,cleared,axis,lanes,position,
    Gemm.Both{%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shape_exact) :
      Global.Shaped(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~value),target,_,cleared) restated,
      %Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shape_exact) : Public.Laid<_,axis,lanes,position,values> laid})

def sealed_observation(~Logical: Data,~value: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+shape: List<&2,U32>,
  +limits: Machine.Limits,+axis: Nat,+lanes: U32,+position: Grid.Position,+workers: U32,+padding_layout: Padded.Layout,
  +positive: {Nat.is_lt(0n,Padded.layout_total(padding_layout)) == True{} : Bool},
  +layout_limits: {PiecePadding.layout_limits(padding_layout) == limits : Machine.Limits},
  +total: {Padded.layout_total(padding_layout) == Shape.elements(shape) : Nat},
  +shape_exact: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +laid: Public.Laid<shape,axis,lanes,position,values>,
  +room: {Nat.is_le(Shape.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +reader: Placed.Reader,-pieces: Pieces.Pieces(~Tile.Tile8),+seen: Model.Observation<Tile.Tile8,pieces>,
  +raw: Global.Contract(~Tile.Tile8,~Placed.Reader,~Placed.read,reader,Tensors.Tensor{pieces,shape,Products.zero(),Tensors.Usable{},workers}),
  agrees: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),Shape.elements(shape)) == True{} : Bool} ->
    {cell_mask(~Placed.Reader,~Placed.read,Padded.layout_padding(padding_layout),reader,word) ==
      Representation.lifted(~Logical,~value,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word) : Tile.Tile8}) ->
  Public.Contract(~Logical,~value,logical,values,TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{pieces,shape,Products.zero(),Tensors.Usable{},workers},axis,lanes,position},
    Padded.layout_padding(padding_layout)))):
  Model.Observation{+model,+exact} = seen
  +padding = Padded.layout_padding(padding_layout)
  +half = Machine.half_word(limits)
  +scope = {Restore.Scope{reader,padding,half,Limits.half_word_fits(limits),
    %layout_limits : {PiecePadding.groups_fit(Machine.half_word(_),padding) == True{} : Bool}
    PiecePadding.layout_fit(padding_layout,positive)} : Restore.Scope<Placed.Reader>}
  %exact : Public.Contract(~Logical,~value,logical,values,TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{_,shape,Products.zero(),Tensors.Usable{},workers},axis,lanes,position},padding)))
  sealed(~Logical,~value,logical,values,shape,axis,lanes,position,workers,model,reader,scope,{==},shape_exact,laid,
    Equal.trans(Nat,PiecePadding.padded_cells(padding),Padded.layout_total(padding_layout),Shape.elements(shape),PiecePadding.layout_cells(padding_layout),total),
    half,Limits.within_half(limits,Shape.elements(shape),room),
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,model),pieces,exact) :
      Global.Contract(~Tile.Tile8,~Placed.Reader,~Placed.read,reader,Tensors.Tensor{_,shape,Products.zero(),Tensors.Usable{},workers})
    raw,agrees)

def plan_contract(~Logical: Data,~value: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+shape: List<&2,U32>,
  +limits: Machine.Limits,+axis: Nat,+lanes: U32,+position: Grid.Position,+workers: U32,+padding_layout: Padded.Layout,
  +positive: {Nat.is_lt(0n,Padded.layout_total(padding_layout)) == True{} : Bool},
  +layout_limits: {PiecePadding.layout_limits(padding_layout) == limits : Machine.Limits},
  +total: {Padded.layout_total(padding_layout) == Shape.elements(shape) : Nat},
  +shape_exact: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +laid: Public.Laid<shape,axis,lanes,position,values>,
  +room: {Nat.is_le(Shape.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +unit: Nat,+grid_layout: Grids.GridLayout<shape,Grid.piece_target(),unit>,
  +plan: Placed.shape_plan(limits,Pieces.ranges_of_grid(Tensors.grid_of(shape))),
  +correct: Masked(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~value),Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
    Padded.layout_padding(padding_layout),limits,Pieces.ranges_of_grid(Tensors.grid_of(shape)),
    Runs.range_low(Pieces.ranges_of_grid(Tensors.grid_of(shape))),Runs.range_high(Pieces.ranges_of_grid(Tensors.grid_of(shape))),plan),
  pieces: Pieces.Pieces(~Tile.Tile8),+same: {pieces == Placed.installed(limits,Pieces.ranges_of_grid(Tensors.grid_of(shape)),plan) : Pieces.Pieces(~Tile.Tile8)}) ->
  Public.Contract(~Logical,~value,logical,values,TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{pieces,shape,Products.zero(),Tensors.Usable{},workers},axis,lanes,position},
    Padded.layout_padding(padding_layout)))):
  Grids.GridLayout{+run,+low,+high} = grid_layout
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +first = Runs.range_low(layout)
  +last = Runs.range_high(layout)
  +reader = Placed.planned_reader(limits,layout,first,last,plan)
  +target = {Representation.Lifted{logical,Public.geometry_of(values,axis,position)} : Representation.Lifted<Logical>}
  sealed_observation(~Logical,~value,logical,values,shape,limits,axis,lanes,position,workers,padding_layout,positive,layout_limits,total,shape_exact,laid,room,reader,
    pieces,Model.observe(~Tile.Tile8,pieces),Placed.shape_contract(limits,shape,workers,unit,room,default_capacity,grid_layout,plan,pieces,same),
    word => inside => masked_cell(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~value),target,Padded.layout_padding(padding_layout),limits,layout,first,last,
      plan,correct,word,%Equal.sym(Nat,first,0n,low) : {Nat.is_le(_,U32.to_nat(word)) == True{} : Bool} Order.zero_le(U32.to_nat(word)),
      %Equal.sym(Nat,last,Shape.elements(shape),high) : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} inside))

