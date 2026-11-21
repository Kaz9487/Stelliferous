# Shared rectangles hold the cells of the public product at their output indices.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_point_products.bend as Products
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/kernel_gemm_point_jobs.bend as Jobs
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_halo_windows.bend as Halo
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_result.bend as Result
import ../../lib/proofs/tensor_point_product_tasks.bend as Shared
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/tensor_shared_product_placement.bend as Placement
import ../../lib/proofs/tensor_shared_product_samples.bend as Span
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_tiled_padded_cells.bend as Padded
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ./product_values.bend as Logical
import ./shared_point_product_lanes.bend as Lanes

type Leaf<-Left: Data,-left_value: Left -> U32 -> F32,-Weight: Data,-weight_value: Weight -> U32 -> F32,-Bias: Data,-bias_value: Bias -> U32 -> F32> is Data:
  Leaf{limits: Machine.Limits,left: Left,weight: Weight,bias: Bias,rows: U32,inner: U32,channels: U32,grid: Grid.Grid,
    first: U32,count: U32,holding: Held.Holding<Tile.Tile8>,whole: Lanes.WholeOperands<Left,left_value,Bias,bias_value,left,bias,rows,inner,channels>,
    span: Span.Span<rows,inner,channels,first,count>,clip: Shared.Clipped<limits,rows,channels,first,count>,
    window: Halo.TaskWindow<Tile.Tile8,Representation.Lifted<Weight>,Representation.lifted(~Weight,~weight_value),
      Representation.Lifted{weight,Samples.shared_geometry(inner,channels)},holding,Span.start_word(rows,inner,first),
      (Span.start_word(rows,inner,first) + Span.span_word(rows,inner,count) : U32),Span.weight_cells(inner,channels)>,
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(inner))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    weight_room: {Nat.is_le(Span.weight_cells(inner,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    bias_room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(Nat.mul(1n,U32.to_nat(rows)),U32.to_nat(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    input_groups: TiledValues.Groups<limits,inner>,
    run_room: Lanes.RunRoom<limits,rows,inner,Shared.clipped_extent(rows,channels,first,count)>,
    complete: {U32.is_zero((inner % 16 : U32)) == True{} : Bool},
    aligned: {U32.to_nat(first) == Nat.mul(Span.start_group(rows,first),U32.to_nat(rows)) : Nat},
    read_room: Fragments.RectangleReadRoom<limits,Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)>,
    layout: Padded.Layout,geometry_exact: {Padded.layout_geometry(layout) == Span.output_geometry(rows,channels) : Cells.Cells},
    covered: {Nat.is_le(Nat.add(U32.to_nat(first),Fragments.rectangle_cells(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count))),
      Padded.layout_total(layout)) == True{} : Bool}}

def source_of(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Fragments.RectangleRead:
  Leaf{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,
    Lanes.WholeOperands{+input,+offset,input_length,input_values,offset_length,offset_values},
    span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,run_room,complete,aligned,read_room,layout,geometry_exact,covered} = leaf
  Fragments.RectangleRead{Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),
    Buffer.content_tree(~Tile.Tile16,Placement.leaf_stage(rows,inner,channels,grid,first,count,holding,Point.SharedResource{input,offset}))}

def expected_of(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Logical.Shared<Left,Weight,Bias>:
  Leaf{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,layout,geometry_exact,covered} = leaf
  Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)}

def rectangle_of(+source: Fragments.RectangleRead) -> Products.Rectangle:
  Fragments.RectangleRead{rectangle,stage} = source
  rectangle

def geometry_of(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Cells.Cells:
  Leaf{limits,left,weight,bias,+rows,inner,+channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,layout,geometry_exact,covered} = leaf
  Span.output_geometry(rows,channels)

type Cell<-Left: Data,-left_value: Left -> U32 -> F32,-Weight: Data,-weight_value: Weight -> U32 -> F32,-Bias: Data,-bias_value: Bias -> U32 -> F32> is Data:
  Cell{leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>,word: U32,
    local: Fragments.Coordinate<rectangle_of(source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)),word>}

def group_of(+rectangle: Products.Rectangle,+rows: U32,+first: U32,+word: U32) -> Nat:
  Nat.add(Span.start_group(rows,first),Fragments.rectangle_group(rectangle,word))

def LaneValue(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>,+lane: Nat) -> Data:
  Cell{+leaf,+word,local} = cell
  Leaf{limits,left,weight,bias,+rows,inner,channels,grid,+first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,layout,geometry_exact,covered} = leaf
  +source = source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  +rectangle = rectangle_of(source)
  +position = Fragments.rectangle_position(rectangle,word)
  Result.MaskedLane(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    expected_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),
    geometry_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),position,group_of(rectangle,rows,first,word),Fragments.cell_value(source,word),lane)

def lane_extent(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Nat:
  8n

def cell_lane(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>,+lane: Nat,
  +inside: {Nat.is_lt(lane,lane_extent(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,cell)) == True{} : Bool}) ->
  LaneValue(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,cell,lane):
  Cell{+leaf,+word,+coordinate} = cell
  Leaf{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,Lanes.WholeOperands{+input,+offset,+input_length,+input_values,+offset_length,+offset_values},+span,+clip,+window,+input_room,+weight_room,+bias_room,+output_room,+input_groups,
    +run_room,+complete,+aligned,+read_room,+layout,+geometry_exact,+covered} = leaf
  Fragments.Coordinate{+position_inside,+group_inside,+index_exact} = coordinate
  +geometry = Span.output_geometry(rows,channels)
  +source = source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  +position = Fragments.rectangle_position(rectangle,word)
  +group = Fragments.rectangle_group(rectangle,word)
  +global_group = group_of(rectangle,rows,first,word)
  +channel = Nat.add(lane,Nat.mul(global_group,8n))
  +logical = {Logical.Shared{left,weight,bias,U32.to_nat(inner),U32.to_nat(channels)} : Logical.Shared<Left,Weight,Bias>}
  +tile = Fragments.cell_value(source,word)
  Result.masked_lane(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    logical,geometry,position,global_group,tile,lane,Order.strict_follow(lane,8n,inside),
    active =>
      +args = {Lanes.LaneOperands{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,Lanes.WholeOperands{input,offset,input_length,input_values,offset_length,offset_values},span,clip,window,input_room,weight_room,bias_room,input_groups,
        position,group,lane,position_inside,group_inside,inside,active} : Lanes.LaneOperands<Left,left_value,Weight,weight_value,Bias,bias_value>}
      +normal = Nat.add(Nat.mul(position,U32.to_nat(channels)),channel)
      %Equal.sym(Nat,Representation.logical_index(geometry,position,channel),normal,Representation.last_axis_index(geometry,position,channel,{==})) :
        {TileLaws.select_lane(lane,tile) == Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,logical,U32.from_nat(_)) : F32}
      Lanes.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,args,TileLaws.select_lane(lane,tile),
        Lanes.staged_dot(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,args,run_room,complete),output_room))

def leaf_length(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Nat:
  Fragments.rectangle_cells(rectangle_of(source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)))

def At(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>,+index: Nat) -> Data:
  Leaf{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,+layout,geometry_exact,covered} = leaf
  Result.At(~Representation.Lifted<Logical.Shared<Left,Weight,Bias>>,~Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value)),
    Representation.Lifted{expected_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),geometry_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)},
    Padded.layout_padding(layout),source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),index)

def CellInside(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  Cell{Leaf{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,+layout,geometry_exact,covered},+word,coords} = cell
  {Nat.is_lt(U32.to_nat(word),Padded.layout_total(layout)) == True{} : Bool}

def TileValue(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>) -> Data:
  Cell{+leaf,+word,coords} = cell
  Leaf{limits,left,weight,bias,rows,inner,channels,grid,first,count,holding,whole,span,clip,window,input_room,weight_room,bias_room,output_room,input_groups,
    run_room,complete,aligned,read_room,+layout,geometry_exact,covered} = leaf
  {Result.cell_mask(~Fragments.RectangleRead,~Fragments.cell_value,Padded.layout_padding(layout),source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),word) ==
    Representation.lifted(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
      Representation.Lifted{expected_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),geometry_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)},word) : Tile.Tile8}

def stored_tile(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +cell: Cell<Left,left_value,Weight,weight_value,Bias,bias_value>,+inside: CellInside(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,cell)) ->
  TileValue(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,cell):
  Cell{+leaf,+word,+coords} = cell
  Leaf{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,Lanes.WholeOperands{+input,+offset,+input_length,+input_values,+offset_length,+offset_values},
    Span.Span{+group_cover,start_exact,length_exact,window_inside},Shared.Clipped{clip_exact,clip_room,clip_positive,+clip_groups},+window,
    +input_room,+weight_room,+bias_room,+output_room,+input_groups,+run_room,+complete,+aligned,+read_room,+layout,+geometry_exact,+covered} = leaf
  Fragments.Coordinate{+position_inside,+group_inside,+local_exact} = coords
  +source = source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  +rectangle = rectangle_of(source)
  +position = Fragments.rectangle_position(rectangle,word)
  +group = Fragments.rectangle_group(rectangle,word)
  +global_group = group_of(rectangle,rows,first,word)
  +local_group = {%clip_groups : {Nat.is_lt(group,_) == True{} : Bool} group_inside :
    {Nat.is_lt(group,Span.group_count(rows,count)) == True{} : Bool}}
  +whole_group = Lanes.selected_group(Span.start_group(rows,first),Span.group_count(rows,count),group,U32.to_nat(Gemm.blocks(channels)),local_group,group_cover)
  +geometry = Span.output_geometry(rows,channels)
  +tile = Fragments.cell_value(source,word)
  +expected = expected_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  +valid = Result.valid_tile(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    expected,geometry,position,global_group,tile,
    Members.range_all(~Cell<Left,left_value,Weight,weight_value,Bias,bias_value>,
      ~LaneValue(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),~lane_extent(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
      ~cell_lane(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),8n,0n,cell,{==}))
  Result.masked_decoded(~Logical.Shared<Left,Weight,Bias>,~Logical.shared_value(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    expected,layout,geometry,position,global_group,word,tile,geometry_exact,
    Span.output_index(rows,inner,channels,grid,first,count,word,aligned,coords),inside,position_inside,whole_group,valid)

def leaf_cell(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>,+index: Nat,
  +inside: {Nat.is_lt(index,leaf_length(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)) == True{} : Bool}) ->
  At(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf,index):
  Leaf{+limits,+left,+weight,+bias,+rows,+inner,+channels,+grid,+first,+count,+holding,Lanes.WholeOperands{+input,+offset,+input_length,+input_values,+offset_length,+offset_values},+span,+clip,+window,+input_room,+weight_room,+bias_room,+output_room,+input_groups,
    +run_room,+complete,+aligned,+read_room,+layout,+geometry_exact,+covered} = leaf
  +source = source_of(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  +rectangle = rectangle_of(source)
  +start = U32.to_nat(first)
  +extent = Fragments.rectangle_cells(rectangle)
  +end = Nat.add(start,extent)
  +position = Nat.add(start,index)
  +word = U32.from_nat(position)
  +cap = Machine.capacity(limits)
  +before = Order.offset_lt(start,index,extent,inside)
  +bounded = Bounds.strict_le(position,U32.to_nat(cap),Order.lt_le(position,end,U32.to_nat(cap),before,Fragments.rectangle_room(limits,rectangle,read_room)))
  +exact = Bounds.roundtrip(position,cap,Limits.capacity_fits(limits),bounded)
  +coords = Fragments.rectangle_coordinate(limits,rectangle,word,read_room,
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_le(start,_) == True{} : Bool} Order.le_add(start,index),
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_lt(_,end) == True{} : Bool} before)
  stored_tile(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,Cell{leaf,word,coords},
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_lt(_,Padded.layout_total(layout)) == True{} : Bool}
    Order.lt_le(position,end,Padded.layout_total(layout),before,covered))

def leaf_values(~Left: Data,~left_value: Left -> U32 -> F32,~Weight: Data,~weight_value: Weight -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +leaf: Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>) ->
  Members.All(Nat,position => At(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf,position),
    Ranges.range(leaf_length(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf),0n)):
  +length = leaf_length(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value,leaf)
  Members.range_all(~Leaf<Left,left_value,Weight,weight_value,Bias,bias_value>,~At(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    ~leaf_length(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),~leaf_cell(~Left,~left_value,~Weight,~weight_value,~Bias,~bias_value),
    length,0n,leaf,Order.reflexive(length))
