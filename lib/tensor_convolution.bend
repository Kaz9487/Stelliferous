# CHW / NCHW entry. Geometry and allocation checks finish before planning.
# The input has its lanes on the channels with the groups innermost, the
# weights their lanes on the output channels with the groups outermost, so
# that their cells are the lane tiles of the product. The output pieces are
# those of the output cells.
import Base
import ./tensor.bend as Tensors
import ./storage_buffer.bend as Storage
import ./kernel_shape.bend as KernelShape
import ./kernel_tile.bend as Tile
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./proofs/storage_certificate.bend as Certificate
import ./kernel_window_pipeline.bend as WindowPipeline
import ./kernel_pipeline.bend as Pipeline
import ./math_activation.bend as Activations
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_tiled.bend as Tiled

type InputShape is Data:
  InputShape{batch: U32,batched: Bool,channels: U32,height: U32,width: U32}

type Geometry is Data:
  Geometry{batch: U32,batched: Bool,dimensions: KernelShape.Dimensions}

def input_shape(shape: List<&2,U32>) -> Maybe<&2,InputShape>:
  match shape:
    case [channels,height,width]: Some{InputShape{1,False{},channels,height,width}}
    case [batch,channels,height,width]: Some{InputShape{batch,True{},channels,height,width}}
    case shape: None{}

def matched(matches: Bool,geometry: Geometry) -> Maybe<&2,Geometry>:
  match matches:
    case True{}: Some{geometry}
    case False{}: None{}

def geometry(input: Maybe<&2,InputShape>,weights: List<&2,U32>,bias: List<&2,U32>,stride: U32,padding: U32) -> Maybe<&2,Geometry>:
  match input weights bias:
    case Some{InputShape{batch,batched,+channels,height,width}} Con{+rows,Con{weight_channels,Con{+kernel,Con{kernel_width,Nil{}}}}} Con{bias_channels,Nil{}}:
      matched(U32.is_eq(channels,weight_channels) && U32.is_eq(kernel,kernel_width) && U32.is_eq(rows,bias_channels),
        Geometry{batch,batched,KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}})
    case input weights bias: None{}

def output_shape(batched: Bool,batch: U32,rows: U32,height: U32,width: U32) -> List<&2,U32>:
  match batched:
    case False{}: [rows,height,width]
    case True{}: [batch,rows,height,width]

# The cells of an image batch: the positions of every image, the groups of a
# position side by side.
def image_cells(batched: Bool,+batch: U32,+height: U32,+width: U32,+groups: U32) -> List<&2,U32>:
  match batched:
    case False{}: [height,width,groups]
    case True{}: [batch,height,width,groups]

# The lane axis of an input or output of the call: its channels.
def lane_of(batched: Bool) -> Nat:
  match batched:
    case False{}: 0n
    case True{}: 1n

def machine_support(geometry: Geometry,+limits: Machine.Limits) -> Bool:
  Geometry{+count,batched,+dimensions} = geometry
  KernelShape.Dimensions{+ci,+rows,+h,+w,+k,+s,+p} = dimensions
  KernelShape.supports_packed(limits,dimensions,Nat.mul(ci,Nat.mul(h,w)),Nat.mul(rows,Nat.mul(ci,Nat.mul(k,k)))) &&
    Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(rows,Nat.mul(KernelShape.axis_size(h,k,s,p),KernelShape.axis_size(w,k,s,p)))),U32.to_nat(Machine.capacity(limits)))

def prefix_of(grid: Grid.Grid) -> U32:
  Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  prefix

def images_of(+grid: Grid.Grid,+image: U32) -> U32:
  Grid.Grid{count,prefix,outer,extent,+unit_cells,+piece_units,per_outer,pieces} = grid
  ((unit_cells * piece_units) / U32.max(image,1) : U32)

# When the pieces of both grids hold whole images, a task holds whole pieces
# of both: the output cells of the most images a piece of either holds.
def whole_cells(+lead: U32,+input: Grid.Grid,+output: Grid.Grid,+input_image: U32,+output_image: U32) -> U32:
  Bool.pick(U32,U32.is_le(prefix_of(input),lead) && U32.is_le(prefix_of(output),lead),
    (output_image * U32.max(U32.max(images_of(input,input_image),images_of(output,output_image)),1) : U32),1)

# The packed input tiles of one output cell: its share of the inner extent of
# a position over the output groups, unless the input is read in place.
def packed_share(+convolution: WindowPipeline.Convolution) -> F32:
  WindowPipeline.Convolution{h,w,k,s,p,oh,ow,channels,+inner,gi,+go,activation} = convolution
  Bool.pick(F32,WindowPipeline.pointwise(convolution),0.0,(U32.to_f32(inner) / 8.0 / U32.to_f32(U32.max(go,1)) : F32))

# The pairs of the planes of the whole output, unless the input is read in
# place.
def plane_pairs(+convolution: WindowPipeline.Convolution) -> U32:
  WindowPipeline.Convolution{h,w,k,s,p,+oh,+ow,channels,inner,gi,go,activation} = convolution
  Bool.pick(U32,WindowPipeline.pointwise(convolution),0,WindowPipeline.plane_size(convolution,(oh * ow : U32)))

# The transposed pairs of one output cell: its share of the planes.
def transposed_share(+convolution: WindowPipeline.Convolution) -> F32:
  (U32.to_f32(plane_pairs(convolution)) / U32.to_f32(U32.max(WindowPipeline.output_image_cells(convolution),1)) : F32)

# One output cell: a multiply-add row of its eight channels for every inner
# index, its store, its activation, eight lanes a row, its packed tiles and
# its transposed pairs.
def cell_rows(+convolution: WindowPipeline.Convolution) -> F32:
  WindowPipeline.Convolution{h,w,k,s,p,oh,ow,channels,+inner,gi,go,+activation} = convolution
  (U32.to_f32(inner) + 1.0 + Bool.pick(F32,Activations.is_identity(activation),0.0,Activations.operations(activation))
    + packed_share(convolution) * Pipeline.packed_tile_rows() + transposed_share(convolution) * Pipeline.transposed_pair_rows() : F32)

def cell_steps(+convolution: WindowPipeline.Convolution) -> F32:
  (4.0 + packed_share(convolution) * Pipeline.packed_tile_steps() + transposed_share(convolution) * Pipeline.transposed_pair_steps() : F32)

# What a task beyond the first adds: its own weights, bias and workspace,
# and a halo of the rows a window reaches past the step when tasks split
# images. Its planes hold the pairs of its own positions, so they grow with
# its cells, which the transposed share of a cell counts.
def extra_of(+whole: U32,+convolution: WindowPipeline.Convolution) -> Tasks.Extra:
  WindowPipeline.Convolution{h,+w,+k,+s,p,+oh,+ow,channels,+inner,+gi,+go,activation} = convolution
  +halo = Bool.pick(U32,U32.is_lt(1,whole) || U32.is_le(k,s),0,((k - s) * w * gi : U32))
  Tasks.Extra{(U32.to_f32((go * inner + go + WindowPipeline.workspace_size(convolution,(oh * ow : U32)) : U32)) * 32.0 : F32),(U32.to_f32(halo) * Cost.copy_ns(64.0,DefaultCost.configuration()) : F32),0.0}

# What a cut task pays for each of its output cells: a copy of the input
# cells it reads, which another task owns, and the copy of its cell into the
# piece.
def cut_cell_ns(+convolution: WindowPipeline.Convolution) -> F32:
  ((U32.to_f32(WindowPipeline.image_cells(convolution)) / U32.to_f32(U32.max(WindowPipeline.output_image_cells(convolution),1)) + 1.0)
    * Cost.copy_ns(64.0,DefaultCost.configuration()) : F32)

def run_windowed(~custom: F32 -> F32,+lead: U32,+convolution: WindowPipeline.Convolution,+input_grid: Grid.Grid,+output_grid: Grid.Grid,+rows: F32,+steps: F32,+workers: U32,
  pieces: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  +whole = whole_cells(lead,input_grid,output_grid,WindowPipeline.image_cells(convolution),WindowPipeline.output_image_cells(convolution))
  WindowPipeline.windowed(~custom,convolution,input_grid,output_grid,Tasks.decide_cut(output_grid,workers,whole,rows,steps,cut_cell_ns(convolution),extra_of(whole,convolution)),pieces,weights,bias)

def run(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,pieces: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  +workers: U32) -> Tiled.Tiled:
  Geometry{+count,+batched,KernelShape.Dimensions{+ci,+rows,+height,+width,+kernel,+stride,+padding}} = geometry
  +oh = U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding))
  +ow = U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))
  +channels = U32.from_nat(rows)
  +input_groups = Tiled.groups(U32.from_nat(ci))
  +output_groups = Tiled.groups(channels)
  +inner = U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))
  +input_grid = Tensors.grid_of(image_cells(batched,count,U32.from_nat(height),U32.from_nat(width),input_groups))
  +cells = image_cells(batched,count,oh,ow,output_groups)
  +output_grid = Tensors.grid_of(cells)
  +convolution = {WindowPipeline.Convolution{U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),oh,ow,channels,inner,input_groups,
    output_groups,activation} : WindowPipeline.Convolution}
  Tiled.Tiled{Tensors.Tensor{run_windowed(~custom,Bool.pick(U32,batched,1,0),convolution,input_grid,output_grid,cell_rows(convolution),cell_steps(convolution),workers,pieces,weights,
    bias),cells,Tiled.zero_tile(),Tensors.Usable{},workers},lane_of(batched),channels,
    Grid.Innermost{}}

# A failure of the call keeps the input cells.
def failed(+reason: Tensors.Reason,pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,fill: Tile.Tile8,workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Failed{Tensors.Problem{"conv2d",reason}},workers},axis,lanes,position}

def band_machine(~custom: F32 -> F32,+activation: Activations.Activation,supported: Bool,+geometry: Geometry,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,+shape: List<&2,U32>,+cells: List<&2,U32>,fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position) ->
  Tiled.Tiled:
  match supported:
    case True{}: run(~custom,activation,geometry,pieces,weights,bias,workers)
    case False{}: failed(Tensors.InvalidShape{shape},pieces,cells,fill,workers,axis,lanes,position)

def band_supported(~custom: F32 -> F32,+activation: Activations.Activation,valid: Bool,+geometry: Geometry,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,+shape: List<&2,U32>,+cells: List<&2,U32>,fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position) ->
  Tiled.Tiled:
  match valid:
    case False{}: Tiled.rejected("conv2d",Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},axis,lanes,position})
    case True{}:
      band_machine(~custom,activation,machine_support(geometry,Default.limits()),geometry,pieces,weights,bias,shape,cells,fill,workers,axis,lanes,position)

def parameters_valid(+dimensions: KernelShape.Dimensions,+weight_length: Nat,+weight_depth: Nat,+bias_length: Nat,+bias_depth: Nat) -> Bool:
  KernelShape.Dimensions{+ci,+rows,+h,+w,+k,s,p} = dimensions
  +inner = Nat.mul(ci,Nat.mul(k,k))
  +groups = Nat.div(Nat.add(rows,7n),8n)
  KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),Nat.mul(rows,inner),rows) &&
    Nat.is_eq(weight_length,Nat.mul(groups,inner)) && Nat.is_eq(bias_length,groups) &&
    Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) && Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth)))

def band_checked(~custom: F32 -> F32,+activation: Activations.Activation,geometry: Maybe<&2,Geometry>,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,+shape: List<&2,U32>,+cells: List<&2,U32>,fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position) ->
  Tiled.Tiled:
  match geometry:
    case None{}: Tiled.rejected("conv2d",Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Usable{},workers},axis,lanes,position})
    case Some{+geometry}:
      Geometry{count,batched,+dimensions} = geometry
      Storage.Buffer{weights,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
      Storage.Buffer{bias,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
      band_supported(~custom,activation,parameters_valid(dimensions,weight_length,weight_depth,bias_length,bias_depth),geometry,pieces,
        Storage.Buffer{weights,weight_length,Certificate.Certificate{weight_depth,weight_equation}},
        Storage.Buffer{bias,bias_length,Certificate.Certificate{bias_depth,bias_equation}},shape,cells,fill,workers,axis,lanes,position)

def band_status(~custom: F32 -> F32,+activation: Activations.Activation,status: Tensors.Status,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+cells: List<&2,U32>,+fill: Tile.Tile8,
  weights: Storage.Buffer<Tile.Tile8>,weight_shape: List<&2,U32>,bias: Storage.Buffer<Tile.Tile8>,bias_shape: List<&2,U32>,stride: U32,padding: U32,+workers: U32,+axis: Nat,+lanes: U32,
  +position: Grid.Position) -> Tiled.Tiled:
  match status:
    case Tensors.Failed{problem}: Tiled.Tiled{Tensors.Tensor{pieces,cells,fill,Tensors.Failed{problem},workers},axis,lanes,position}
    case Tensors.Usable{}:
      band_checked(~custom,activation,geometry(input_shape(shape),weight_shape,bias_shape,stride,padding),pieces,weights,bias,shape,cells,fill,workers,axis,lanes,position)

def input_cells(~custom: F32 -> F32,+activation: Activations.Activation,input: Tiled.Tiled,weights: Storage.Buffer<Tile.Tile8>,weight_shape: List<&2,U32>,bias: Storage.Buffer<Tile.Tile8>,bias_shape: List<&2,U32>,
  parameter_status: Tensors.Status,parameter_workers: U32,stride: U32,padding: U32) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.Tensor{pieces,+cells,+fill,input_status,workers},+axis,+lanes,+position} = input
  band_status(~custom,activation,Tensors.first_failure(input_status,parameter_status),pieces,Tiled.logical_shape(cells,axis,lanes,position),cells,fill,weights,weight_shape,bias,bias_shape,
    stride,padding,U32.max(workers,parameter_workers),axis,lanes,position)

# The input with its lanes on the channels, the axis before the rows, and the
# groups innermost.
def channel_lanes(pair: Tiled.Tiled & List<&2,U32>) -> Tiled.Tiled:
  (input,shape) = pair
  Tiled.arranged(input,Tiled.lane_axis(List.length(&2,U32,shape)),Grid.Innermost{})

# The cells of the weights are the lane tiles of the product; the bias is one
# cell per output-channel group.
def conv2d_aligned(~custom: F32 -> F32,+activation: Activations.Activation,input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled,+stride: U32,+padding: U32) -> Tiled.Tiled:
  Tiled.Tiled{weight_cells,+weight_axis,+weight_lanes,+weight_position} = weights
  Tiled.Tiled{bias_cells,+bias_axis,+bias_lanes,+bias_position} = bias
  Tensors.with_whole(~Tile.Tile8,~Tiled.Tiled,weight_cells,weights => weight_shape => weight_fill => weight_status => weight_workers =>
    Tensors.with_whole(~Tile.Tile8,~Tiled.Tiled,bias_cells,bias => bias_shape => bias_fill => bias_status => bias_workers =>
      input_cells(~custom,activation,channel_lanes(Tiled.shape(input)),weights,Tiled.logical_shape(weight_shape,weight_axis,weight_lanes,weight_position),bias,
        Tiled.logical_shape(bias_shape,bias_axis,bias_lanes,bias_position),Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride,padding)))

def conv2d(~custom: F32 -> F32,activation: Activations.Activation,input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled,stride: U32,padding: U32) -> Tiled.Tiled:
  conv2d_aligned(~custom,activation,input,Tiled.arranged(weights,0n,Grid.Outermost{}),Tiled.along(bias,0n),stride,padding)

# The products of count images of rows positions and inner channels with
# outputs rows of inner weights that every image shares, plus the bias: a
# convolution of a one-cell window. The cells of the input are
# [count,rows,1,input groups], those of the result [count,rows,1,output groups].
def products(+count: U32,+rows: U32,+inner: U32,+outputs: U32,pieces: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  +workers: U32) -> Tensors.Tensor<Tile.Tile8>:
  Tiled.cells_of(run(~Activations.identity_value,Activations.identity(),
    Geometry{count,True{},KernelShape.Dimensions{U32.to_nat(inner),U32.to_nat(outputs),U32.to_nat(rows),U32.to_nat(1),U32.to_nat(1),U32.to_nat(1),U32.to_nat(0)}},pieces,weights,bias,workers))
