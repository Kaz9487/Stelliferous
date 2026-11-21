# Read the affected output cells at the logical coordinates of each image.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_point_products.bend as Products
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_gemm_point_jobs.bend as Jobs
import ../../lib/proofs/kernel_tile_refinement.bend as Tiles
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/row_major_index.bend as Index
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_panel_product_samples.bend as Panels
import ../../lib/proofs/tensor_point_product_evaluation.bend as Evaluate
import ../../lib/proofs/tensor_point_product_fragments.bend as Fragments
import ../../lib/proofs/tensor_point_product_result.bend as Result
import ../../lib/proofs/tensor_point_product_samples.bend as Samples
import ../../lib/proofs/tensor_point_product_values.bend as Values
import ../../lib/proofs/tensor_point_products.bend as Point
import ../../lib/proofs/tensor_tiled_padded_cells.bend as Padded
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ./panel_point_product_lanes.bend as Lanes
import ./product_values.bend as Logical

def output_geometry(+images: U32,+rows: U32,+channels: U32) -> Cells.Cells:
  Samples.row_geometry((images * rows : U32),channels)

type OutputLayout<-limits: Machine.Limits,-images: U32,-rows: U32,-channels: U32> is Data:
  OutputLayout{size: Panels.Size<limits,images,rows,channels>,padding: Padded.Layout,
    geometry: {Padded.layout_geometry(padding) == output_geometry(images,rows,channels) : Cells.Cells},
    total: {Padded.layout_total(padding) == Panels.physical_size(images,rows,channels) : Nat}}

type PanelWindow<-Ready: U32 -> Data,-Left: Data,-left_value: Left -> U32 -> F32,-Right: Data,-right_value: Right -> U32 -> F32> is Data:
  PanelWindow{limits: Machine.Limits,left: Left,right: Right,images: U32,rows: U32,inner: U32,channels: U32,grid: Grid.Grid,first: U32,
    input: Held.Holding<Tile.Tile8>,weights: Held.Holding<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>,
    domain: Lanes.Domain<limits,images,rows,inner,channels>,image_inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(images)) == True{} : Bool},
    left_window: Panels.ImageWindow(~Left,~left_value,left,images,rows,inner,first,input),right_window: Panels.ImageWindow(~Right,~right_value,right,images,inner,channels,first,weights),
    bias_values: Lanes.Bias<Buffer.content(~Tile.Tile8,bias),channels>,ready: Ready(inner),prepared: OutputLayout<limits,images,rows,channels>,
    read_room: Fragments.RectangleReadRoom<limits,Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first)>}

def source_of(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +leaf: PanelWindow<Ready,Left,left_value,Right,right_value>) -> Fragments.RectangleRead:
  PanelWindow{limits,left,right,images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,domain,image_inside,left_window,right_window,bias_values,ready,prepared,read_room} = leaf
  Lanes.image_source(~run,Products.Shape{rows,inner,channels,grid},first,input,weights,bias)

def rect_of(reader: Fragments.RectangleRead) -> Products.Rectangle:
  Fragments.RectangleRead{rectangle,tree} = reader
  rectangle

def output_index(+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+word: U32,
  +start: {U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first))) ==
    Nat.mul(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))) : Nat},
  coordinate: Fragments.Coordinate<Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first),word>) ->
  {U32.to_nat(word) == Representation.cell_index(output_geometry(images,rows,channels),
    Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),Fragments.rectangle_position(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first),word)),
    Fragments.rectangle_group(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first),word)) : Nat}:
  Fragments.Coordinate{position_inside,group_inside,+offset_exact} = coordinate
  +rect = Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first)
  +rn = U32.to_nat(rows)
  +before = Nat.mul(U32.to_nat(first),rn)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +start_word = U32.to_nat(Fragments.rectangle_start(rect))
  +position = Fragments.rectangle_position(rect,word)
  +group = Fragments.rectangle_group(rect,word)
  +local = Fragments.rectangle_offset(rect,word)
  +ideal = Nat.add(Nat.mul(position,groups),group)
  +ending = Representation.cell_index(output_geometry(images,rows,channels),Nat.add(before,position),group)
  %Samples.joined_rows(before,position,groups,group) :
    {U32.to_nat(word) == _ : Nat}
  %start : {U32.to_nat(word) == Nat.add(_,ideal) : Nat}
  offset_exact

type OutputBand<-work: Products.Shape,-images: U32,-rows: U32,-channels: U32,-first: U32> is Data:
  OutputBand{start: {U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,first))) == Nat.mul(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))) : Nat},
    covered: {Nat.is_le(Nat.add(U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(work,first))),Fragments.rectangle_cells(Point.panel_geometry(work,first))),
      Panels.physical_size(images,rows,channels)) == True{} : Bool}}

def output_cover(+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,
  size: Panels.Size<limits,images,rows,channels>,band: Panels.ImageBand<images,rows,channels,first>) ->
  OutputBand<Products.Shape{rows,inner,channels,grid},images,rows,channels,first>:
  Panels.Size{+width,outer,image_room,room} = size
  Panels.ImageBand{+start,+contains} = band
  +from = (first * rows * Gemm.blocks(channels) : U32)
  +band_start = Panels.band_start(first,rows,channels)
  +same = Arithmetic.multiply_associative(first,rows,Gemm.blocks(channels))
  OutputBand{
    %Equal.sym(U32,from,band_start,same) : {U32.to_nat(_) == Nat.mul(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))) : Nat} start,
    %Equal.sym(U32,from,band_start,same) : {Nat.is_le(Nat.add(U32.to_nat(_),Panels.image_size(rows,channels)),Panels.physical_size(images,rows,channels)) == True{} : Bool}
    %width : {Nat.is_le(Nat.add(U32.to_nat(band_start),_),Panels.physical_size(images,rows,channels)) == True{} : Bool} contains}

def output_band(+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,
  +inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(images)) == True{} : Bool},+size: Panels.Size<limits,images,rows,channels>) ->
  OutputBand<Products.Shape{rows,inner,channels,grid},images,rows,channels,first>:
  output_cover(limits,images,rows,inner,channels,grid,first,size,Panels.image_band(limits,images,rows,channels,first,inside,size))


def band_start(+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,
  +band: OutputBand<Products.Shape{rows,inner,channels,grid},images,rows,channels,first>) ->
  {U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first))) ==
    Nat.mul(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(Gemm.blocks(channels))) : Nat}:
  OutputBand{+start,covered} = band
  start

def band_covered(+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,
  +band: OutputBand<Products.Shape{rows,inner,channels,grid},images,rows,channels,first>) ->
  {Nat.is_le(Nat.add(U32.to_nat(Fragments.rectangle_start(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first))),
    Fragments.rectangle_cells(Point.panel_geometry(Products.Shape{rows,inner,channels,grid},first))),Panels.physical_size(images,rows,channels)) == True{} : Bool}:
  OutputBand{start,+covered} = band
  covered
type Cell<-run: Point.Run(),-Ready: U32 -> Data,-Left: Data,-left_value: Left -> U32 -> F32,-Right: Data,-right_value: Right -> U32 -> F32> is Data:
  Cell{leaf: PanelWindow<Ready,Left,left_value,Right,right_value>,word: U32,
    at: Fragments.Coordinate<rect_of(source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)),word>}

def lane_extent(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  cell: Cell<run,Ready,Left,left_value,Right,right_value>) -> Nat:
  8n

def AtLane(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +cell: Cell<run,Ready,Left,left_value,Right,right_value>,+lane: Nat) -> Data:
  Cell{+leaf,+word,at} = cell
  PanelWindow{limits,+left,+right,+images,+rows,+inner,+channels,grid,+first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values,ready,prepared,read_room} = leaf
  +source = source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)
  +rect = rect_of(source)
  +position = Fragments.rectangle_position(rect,word)
  +group = Fragments.rectangle_group(rect,word)
  +outer = Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position)
  Result.MaskedLane(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},output_geometry(images,rows,channels),outer,group,Fragments.cell_value(source,word),lane)

def cell_lane(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +cell: Cell<run,Ready,Left,left_value,Right,right_value>,+lane: Nat,+inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  AtLane(~run,~Ready,~Left,~left_value,~Right,~right_value,cell,lane):
  Cell{+leaf,+word,+coordinate} = cell
  PanelWindow{+limits,+left,+right,+images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,+domain,+image_inside,+left_window,+right_window,+bias_values,+ready,prepared,read_room} = leaf
  Fragments.Coordinate{+position_inside,+group_inside,+offset_exact} = coordinate
  +logical = {Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)} : Logical.Batched<Left,Right>}
  +source = source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)
  +rect = rect_of(source)
  +position = Fragments.rectangle_position(rect,word)
  +group = Fragments.rectangle_group(rect,word)
  +outer = Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +geometry = output_geometry(images,rows,channels)
  +tile = Fragments.cell_value(source,word)
  Result.masked_lane(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),logical,geometry,outer,group,tile,lane,
    Order.strict_follow(lane,8n,inside),
    active =>
      +image = {Lanes.PanelImage{limits,left,right,images,rows,inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values} : Lanes.PanelImage<Left,left_value,Right,right_value>}
      +scope = {Lanes.Lane{image,position,group,lane,position_inside,group_inside,inside,active} : Lanes.Lane<Left,left_value,Right,right_value>}
      +normal = Lanes.dot_index(first,rows,channels,position,channel)
      %Equal.sym(Nat,Representation.logical_index(geometry,outer,channel),normal,Representation.last_axis_index(geometry,outer,channel,{==})) :
        {Tiles.select_lane(lane,tile) == Logical.batched_value(~Left,~left_value,~Right,~right_value,logical,U32.from_nat(_)) : F32}
      Equal.trans(F32,Tiles.select_lane(lane,tile),Lanes.rectangle_value(~Left,~left_value,~Right,~right_value,scope),Lanes.public_lane(~Left,~left_value,~Right,~right_value,scope),
        Lanes.stage_lane(~run,~accumulate,~write,~Ready,~order,limits,images,rows,inner,channels,grid,first,input,weights,bias,domain,ready,
          position,group,lane,position_inside,group_inside,inside),Lanes.local_result(~Left,~left_value,~Right,~right_value,scope)))

def size_outer(+limits: Machine.Limits,+images: U32,+rows: U32,+channels: U32,+size: Panels.Size<limits,images,rows,channels>) ->
  {U32.to_nat((images * rows : U32)) == Panels.row_total(images,rows) : Nat}:
  Panels.Size{width,+outer,image_room,room} = size
  outer

def leaf_length(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  leaf: PanelWindow<Ready,Left,left_value,Right,right_value>) -> Nat:
  Fragments.rectangle_cells(rect_of(source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)))

def At(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +leaf: PanelWindow<Ready,Left,left_value,Right,right_value>,+index: Nat) -> Data:
  PanelWindow{limits,+left,+right,+images,+rows,+inner,+channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values,ready,+prepared,read_room} = leaf
  OutputLayout{size,+padding,geometry,total} = prepared
  Result.At(~Representation.Lifted<Logical.Batched<Left,Right>>,~Representation.lifted(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value)),
    Representation.Lifted{Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},output_geometry(images,rows,channels)},
    Padded.layout_padding(padding),source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf),index)

def CellInside(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +cell: Cell<run,Ready,Left,left_value,Right,right_value>) -> Data:
  Cell{+leaf,+word,at} = cell
  PanelWindow{limits,left,right,images,rows,inner,channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values,ready,+prepared,read_room} = leaf
  OutputLayout{size,+padding,geometry,total} = prepared
  {Nat.is_lt(U32.to_nat(word),Padded.layout_total(padding)) == True{} : Bool}

def TileValue(~run: Point.Run(),~Ready: U32 -> Data,~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +cell: Cell<run,Ready,Left,left_value,Right,right_value>) -> Data:
  Cell{+leaf,+word,at} = cell
  PanelWindow{limits,+left,+right,+images,+rows,+inner,+channels,grid,first,input,weights,bias,domain,image_inside,left_window,right_window,bias_values,ready,+prepared,read_room} = leaf
  OutputLayout{size,+padding,geometry,total} = prepared
  {Result.cell_mask(~Fragments.RectangleRead,~Fragments.cell_value,Padded.layout_padding(padding),source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf),word) ==
    Representation.lifted(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
      Representation.Lifted{Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)},output_geometry(images,rows,channels)},word) : Tile.Tile8}

def stored_tile(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +cell: Cell<run,Ready,Left,left_value,Right,right_value>,+inside: CellInside(~run,~Ready,~Left,~left_value,~Right,~right_value,cell)) ->
  TileValue(~run,~Ready,~Left,~left_value,~Right,~right_value,cell):
  Cell{+leaf,+word,+coordinate} = cell
  PanelWindow{+limits,+left,+right,+images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,+domain,+image_inside,+left_window,+right_window,+bias_values,+ready,
    OutputLayout{+size,+padding,+geometry_exact,+total},+read_room} = leaf
  Fragments.Coordinate{+position_inside,+group_inside,+offset_exact} = coordinate
  +source = source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)
  +rect = rect_of(source)
  +position = Fragments.rectangle_position(rect,word)
  +group = Fragments.rectangle_group(rect,word)
  +outer = Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(rows)),position)
  +geometry = output_geometry(images,rows,channels)
  +logical = {Logical.Batched{left,right,U32.to_nat(rows),U32.to_nat(inner),U32.to_nat(channels)} : Logical.Batched<Left,Right>}
  +tile = Fragments.cell_value(source,word)
  +start = band_start(limits,images,rows,inner,channels,grid,first,output_band(limits,images,rows,inner,channels,grid,first,image_inside,size))
  +outer_inside = {%Equal.sym(Nat,U32.to_nat((images * rows : U32)),Panels.row_total(images,rows),size_outer(limits,images,rows,channels,size)) :
      {Nat.is_lt(outer,_) == True{} : Bool}
    Index.radix(U32.to_nat(first),U32.to_nat(images),position,U32.to_nat(rows),image_inside,position_inside) :
    {Nat.is_lt(outer,Representation.outer_count(geometry)) == True{} : Bool}}
  +valid = Result.valid_tile(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),logical,geometry,outer,group,tile,
    Members.range_all(~Cell<run,Ready,Left,left_value,Right,right_value>,~AtLane(~run,~Ready,~Left,~left_value,~Right,~right_value),
      ~lane_extent(~run,~Ready,~Left,~left_value,~Right,~right_value),
      ~cell_lane(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value),8n,0n,cell,{==}))
  Result.masked_decoded(~Logical.Batched<Left,Right>,~Logical.batched_value(~Left,~left_value,~Right,~right_value),
    logical,padding,geometry,outer,group,word,tile,geometry_exact,output_index(images,rows,inner,channels,grid,first,word,start,coordinate),
    inside,outer_inside,group_inside,valid)

def leaf_cell(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +leaf: PanelWindow<Ready,Left,left_value,Right,right_value>,+index: Nat,
  +inside: {Nat.is_lt(index,leaf_length(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)) == True{} : Bool}) ->
  At(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf,index):
  PanelWindow{+limits,+left,+right,+images,+rows,+inner,+channels,+grid,+first,+input,+weights,+bias,+domain,+image_inside,+left_window,+right_window,+bias_values,+ready,
    OutputLayout{+size,+padding,+geometry_exact,+total},+read_room} = leaf
  +source = source_of(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)
  +rect = rect_of(source)
  +start = U32.to_nat(Fragments.rectangle_start(rect))
  +extent = Fragments.rectangle_cells(rect)
  +end = Nat.add(start,extent)
  +position = Nat.add(start,index)
  +word = U32.from_nat(position)
  +cap = Machine.capacity(limits)
  +total_cells = Panels.physical_size(images,rows,channels)
  +before = Order.offset_lt(start,index,extent,inside)
  +bounded = Bounds.strict_le(position,U32.to_nat(cap),Order.lt_le(position,end,U32.to_nat(cap),before,Fragments.rectangle_room(limits,rect,read_room)))
  +exact = Bounds.roundtrip(position,cap,Limits.capacity_fits(limits),bounded)
  +coords = Fragments.rectangle_coordinate(limits,rect,word,read_room,
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_le(start,_) == True{} : Bool} Order.le_add(start,index),
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_lt(_,end) == True{} : Bool} before)
  +covered = band_covered(limits,images,rows,inner,channels,grid,first,output_band(limits,images,rows,inner,channels,grid,first,image_inside,size))
  stored_tile(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value,Cell{leaf,word,coords},
    %Equal.sym(Nat,U32.to_nat(word),position,exact) : {Nat.is_lt(_,Padded.layout_total(padding)) == True{} : Bool}
    %Equal.sym(Nat,Padded.layout_total(padding),total_cells,total) : {Nat.is_lt(position,_) == True{} : Bool}
    Order.lt_le(position,end,total_cells,before,covered))

def leaf_values(~run: Point.Run(),~accumulate: Jobs.Accumulate(),~write: Values.StepWrites(~run,~accumulate),~Ready: U32 -> Data,~order: Evaluate.KernelOrder(~accumulate,~Ready),
  ~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +leaf: PanelWindow<Ready,Left,left_value,Right,right_value>) ->
  Members.All(Nat,position => At(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf,position),
    Ranges.range(leaf_length(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf),0n)):
  +length = leaf_length(~run,~Ready,~Left,~left_value,~Right,~right_value,leaf)
  Members.range_all(~PanelWindow<Ready,Left,left_value,Right,right_value>,~At(~run,~Ready,~Left,~left_value,~Right,~right_value),
    ~leaf_length(~run,~Ready,~Left,~left_value,~Right,~right_value),~leaf_cell(~run,~accumulate,~write,~Ready,~order,~Left,~left_value,~Right,~right_value),
    length,0n,leaf,Order.reflexive(length))
