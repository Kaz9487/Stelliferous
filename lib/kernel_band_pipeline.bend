# Independent row bands keep their output owners. Fork distributes existing
# core/halo owners; only read-only weights and bias need affine clones.
import Base
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./kernel_window.bend as KernelWindow
import ./kernel_pipeline.bend as Pipeline
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./traversal_partition.bend as Partition
import ./math_activation.bend as Activations

type Context is Data:
  Context{window: BandWindow.Geometry,channels: U32,inner: U32,parallel: Bool,activation: Activations.Activation}

# A leaf assembles the rows its outputs read into one contiguous window whose
# first row is a stride multiple, then runs the leaf on it: packing a
# window is packing a shorter image with output rows shifted by first / stride.
def packed_window(~custom: F32 -> F32,activation: Activations.Activation,+geometry: BandWindow.Geometry,+channels: U32,+inner: U32,+first: U32,+rows: U32,+output_first: U32,+output_rows: U32,
  +window: BandWindow.Rows,+above_range: BandWindow.Rows,+below_range: BandWindow.Rows,
  core: Storage.Buffer<F32>,above: Storage.Buffer<F32>,below: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  BandWindow.Geometry{Bands.Geometry{batch,+ci,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} = geometry
  BandWindow.Rows{+window_first,+window_rows} = window
  +positions = (output_rows * output_width : U32)
  Pipeline.blocked(~custom,activation,
    Pipeline.Work{channels,window_rows,width,kernel,stride,padding,output_width,positions,inner},
    KernelWindow.start(((output_first - first / height * output_height) * output_width : U32),stride,output_width,(window_first - first / height * height : U32)),
    BandWindow.assemble(~F32,0.0,BandWindow.input_geometry(geometry),window,BandWindow.Rows{first,rows},above_range,below_range,core,above,below),weights,bias)

# A window is empty when the leaf has no outputs or its outputs read only
# padding above the image. The leaf then packs the one row at the stride
# position of its first output, which no output reads.
def core_row(+geometry: BandWindow.Geometry,+first: U32,+output_first: U32) -> BandWindow.Rows:
  BandWindow.Geometry{Bands.Geometry{batch,ci,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  BandWindow.Rows{(first / height * height + (output_first - first / height * output_height) * stride : U32),1}

def source_rows(empty: Bool,+geometry: BandWindow.Geometry,+first: U32,+output_first: U32,window: BandWindow.Rows) -> BandWindow.Rows:
  match empty:
    case False{}: window
    case True{}: core_row(geometry,first,output_first)

def windowed(~custom: F32 -> F32,activation: Activations.Activation,+geometry: BandWindow.Geometry,+channels: U32,+inner: U32,+first: U32,+rows: U32,+rows_read: BandWindow.RowWindow,
  core: Storage.Buffer<F32>,above: Storage.Buffer<F32>,below: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Bands.Bands(~F32):
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},+above_range,+below_range,BandWindow.Rows{+window_first,+window_rows}} = rows_read
  Partition.Leaf{packed_window(~custom,activation,geometry,channels,inner,first,rows,output_first,output_rows,
    source_rows(U32.is_zero(window_rows),geometry,first,output_first,BandWindow.Rows{window_first,window_rows}),
    above_range,below_range,core,above,below,weights,bias),output_first,output_rows,Unit{}}

def leaf(~custom: F32 -> F32,context: Context,+first: U32,+rows: U32,core: Storage.Buffer<F32>,halo: BandWindow.Halo<F32>,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Bands.Bands(~F32):
  Context{+geometry,+channels,+inner,parallel,activation} = context
  BandWindow.Halo{above,below,+rows_read} = halo
  windowed(~custom,activation,geometry,channels,inner,first,rows,rows_read,core,above,below,weights,bias)

def parallel(context: Context) -> Bool:
  Context{window,channels,inner,parallel,activation} = context
  parallel

def Parameters() -> Type:
  Storage.Buffer<Tile.Tile8> & Storage.Buffer<F32>

def shared(weights: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Parameters() & Parameters():
  (left_weights,right_weights) = weights
  (left_bias,right_bias) = bias
  ((left_weights,left_bias),(right_weights,right_bias))

# Each half of a split owns its copy of the read-only weights and bias.
def share(parameters: Parameters()) -> Parameters() & Parameters():
  (weights,bias) = parameters
  shared(Clone.clone(~Tile.Tile8,weights),Clone.clone(~F32,bias))

def band(~custom: F32 -> F32,context: Context,first: U32,rows: U32,core: Storage.Buffer<F32>,halo: BandWindow.Halo<F32>,parameters: Parameters()) -> Bands.Bands(~F32):
  (weights,bias) = parameters
  +first = first
  +rows = rows
  leaf(~custom,context,first,rows,core,halo,weights,bias)

# Every core leaf with its halo, through the shared band recursion. Trees of
# another shape keep the cores.
def run(~custom: F32 -> F32,plan: Bands.Plan,+context: Context,bands: Bands.Bands(~F32),halos: BandWindow.Halos(~F32),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Bands.Bands(~F32):
  Bands.each(~Bands.Paired<Storage.Buffer<F32>,BandWindow.Halo<F32>>,~Storage.Buffer<F32>,~Parameters(),~Context,~share,
    ~Bands.paired(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,~Parameters(),~Context,~band(~custom)),
    ~Bands.left_of(~Storage.Buffer<F32>,~BandWindow.Halo<F32>),
    plan,parallel(context),context,Bands.zip(~Storage.Buffer<F32>,~BandWindow.Halo<F32>,bands,halos),(weights,bias))

def prepared(~custom: F32 -> F32,+context: Context,+plan: Bands.Plan,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  owners: Bands.Bands(~F32) & BandWindow.Halos(~F32)) -> Bands.Bands(~F32):
  (bands,halos) = owners
  run(~custom,plan,context,bands,halos,weights,bias)

def planned(~custom: F32 -> F32,+context: Context,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  owners: Bands.Bands(~F32) & Bands.Plan) -> Bands.Bands(~F32):
  Context{window,channels,inner,parallel,activation} = context
  (bands,+plan) = owners
  prepared(~custom,context,plan,weights,bias,BandWindow.prepare(~F32,window,0.0,plan,bands))

# The weights arrive in the order the kernel reads them.
def execute(~custom: F32 -> F32,context: Context,bands: Bands.Bands(~F32),weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Bands.Bands(~F32):
  planned(~custom,context,weights,bias,Bands.plan(~F32,bands))
