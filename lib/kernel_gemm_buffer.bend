import Base
import ./kernel_tile.bend as Tile
import ./kernel_gemm.bend as Gemm
import ./storage_buffer.bend as Storage
import ./proofs/kernel_gemm_certificate.bend as GemmCertificate
import ./traversal_array.bend as Traversal
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/traversal_witness.bend as TraversalWitness

# The row tiles, the lane tiles and the bias come back with the output, for
# a caller that uses them again.
type Kept<-E: Data> is Type:
  Kept{row_tiles: Storage.Buffer<Tile.Tile8>,lane_tiles: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<E>}

def kept(~E: Data,operands: Gemm.Operands<E,Array<F32>>,row_length: Nat,lane_length: Nat,bias_length: Nat,length: Nat,certificates: GemmCertificate.Certified<E,operands>) -> Kept<E>:
  Gemm.Operands{row_tiles,lane_tiles,bias,output} = operands
  GemmCertificate.Certified{row_certificate,lane_certificate,bias_certificate,certificate} = certificates
  Kept{Storage.Buffer{row_tiles,row_length,row_certificate},Storage.Buffer{lane_tiles,lane_length,lane_certificate},
    Storage.Buffer{bias,bias_length,bias_certificate},Storage.Buffer{output,length,certificate}}

# Jobs 0 to count - 1 of a plan. The live certificates are updated once,
# outside every output and MAC loop.
def run(~Plan: Data,~Cursor: Data,~E: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<E,Array<F32>> -> Gemm.Written<E,Array<F32>>,~keeps: GemmCertificate.Keeps(~E,~write),
  +count: Nat,+plan: Plan,row_tiles: Storage.Buffer<Tile.Tile8>,lane_tiles: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<E>) -> Kept<E>:
  Storage.Buffer{row_tiles,row_length,+row_certificate} = row_tiles
  Storage.Buffer{lane_tiles,lane_length,+lane_certificate} = lane_tiles
  Storage.Buffer{bias,bias_length,+bias_certificate} = bias
  Storage.Buffer{output,length,certificate} = output
  kept(~E,Gemm.run(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,count,plan,Gemm.Operands{row_tiles,lane_tiles,bias,output}),
    row_length,lane_length,bias_length,length,
    GemmCertificate.certificates(~Plan,~Cursor,~E,~job_at,~advance,~times,~write,~keeps,count,plan,row_tiles,lane_tiles,bias,output,
      row_certificate,lane_certificate,bias_certificate,certificate))

# A tile product whose epilogue reads cell biases directly.
type CellKept is Type:
  CellKept{row_tiles: Storage.Buffer<Tile.Tile8>,lane_tiles: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

def cells_kept(state: Gemm.CellOperands,row_length: Nat,lane_length: Nat,bias_length: Nat,length: Nat,certificates: GemmCertificate.CellCertified<state>) -> CellKept:
  Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}} = state
  GemmCertificate.CellCertified{bias_certificate,row_certificate,lane_certificate,output_certificate} = certificates
  CellKept{Storage.Buffer{row_tiles,row_length,row_certificate},Storage.Buffer{lane_tiles,lane_length,lane_certificate},
    Storage.Buffer{bias,bias_length,bias_certificate},Storage.Buffer{output,length,output_certificate}}

def cell_keeping(+channels: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  output: Storage.Buffer<Tile.Tile8>,+positions: U32,+inner: U32,+start: U32,+output_positions: U32) -> CellKept:
  Storage.Buffer{row_tiles,row_length,+row_certificate} = input
  Storage.Buffer{lane_tiles,lane_length,+lane_certificate} = weights
  Storage.Buffer{bias,bias_length,+bias_certificate} = bias
  Storage.Buffer{output,length,output_certificate} = output
  +matrix = {Gemm.TileMatrix{channels,positions,inner,start,output_positions} : Gemm.TileMatrix}
  +count = U32.to_nat(Gemm.tile_jobs(matrix))
  cells_kept(Gemm.cell_run(count,matrix,Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,Unit{},output}}),row_length,lane_length,bias_length,length,
    GemmCertificate.cell_certificates(count,matrix,row_tiles,lane_tiles,bias,Unit{},output,row_certificate,lane_certificate,bias_certificate,output_certificate))

# A pointwise product whose rows are read from the input cells, sixteen
# channels a block, written to staged pairs.
type PointKept is Type:
  PointKept{bias: Storage.Buffer<Tile.Tile8>,cells: Storage.Buffer<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>,staged: Storage.Buffer<Tile.Tile16>}

def points_kept(state: Gemm.PointOperands,bias_length: Nat,cell_length: Nat,pair_length: Nat,length: Nat,certificates: GemmCertificate.PointCertified<state>) -> PointKept:
  Gemm.PointOperands{bias,cells,pairs,staged} = state
  GemmCertificate.PointCertified{bias_certificate,cell_certificate,pair_certificate,staged_certificate} = certificates
  PointKept{Storage.Buffer{bias,bias_length,bias_certificate},Storage.Buffer{cells,cell_length,cell_certificate},Storage.Buffer{pairs,pair_length,pair_certificate},
    Storage.Buffer{staged,length,staged_certificate}}

def point_keeping_with(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,~keeps: GemmCertificate.PointJob(~job),
  +matrix: Gemm.PointMatrix,input: Storage.Buffer<Tile.Tile8>,pair_buffer: Storage.Buffer<Tile.Tile16>,bias_buffer: Storage.Buffer<Tile.Tile8>,
  staged_buffer: Storage.Buffer<Tile.Tile16>) -> PointKept:
  Storage.Buffer{cells,cell_length,+cell_certificate} = input
  Storage.Buffer{pairs,pair_length,+pair_certificate} = pair_buffer
  Storage.Buffer{bias,bias_length,+bias_certificate} = bias_buffer
  Storage.Buffer{staged,length,staged_certificate} = staged_buffer
  +count = U32.to_nat(Gemm.point_count(matrix))
  points_kept(Gemm.point_run_indexed(~job,count,matrix,Gemm.PointOperands{bias,cells,pairs,staged}),bias_length,cell_length,pair_length,length,
    GemmCertificate.point_certificates_with(~job,~keeps,count,matrix,bias,cells,pairs,staged,bias_certificate,cell_certificate,pair_certificate,staged_certificate))

def point_keeping(+matrix: Gemm.PointMatrix,input: Storage.Buffer<Tile.Tile8>,pair_buffer: Storage.Buffer<Tile.Tile16>,bias_buffer: Storage.Buffer<Tile.Tile8>,
  staged_buffer: Storage.Buffer<Tile.Tile16>) -> PointKept:
  point_keeping_with(~Gemm.point_job,~GemmCertificate.point_job_whole,matrix,input,pair_buffer,bias_buffer,staged_buffer)

def point_keeping_tail(+matrix: Gemm.PointMatrix,input: Storage.Buffer<Tile.Tile8>,pair_buffer: Storage.Buffer<Tile.Tile16>,bias_buffer: Storage.Buffer<Tile.Tile8>,
  staged_buffer: Storage.Buffer<Tile.Tile16>) -> PointKept:
  point_keeping_with(~Gemm.point_job_with(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner),~GemmCertificate.point_job(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner,~GemmCertificate.finish_inner),matrix,input,pair_buffer,bias_buffer,staged_buffer)

def read_buffers(~R: Data,~E: Data,+length: Nat,+output_length: Nat,result: Array<R> & Array<E>,
  certificates: TraversalCertificate.Certificates<Array<R>,E,Certificate.family(~R),result>) -> Storage.Buffer<R> & Storage.Buffer<E>:
  (reader,output) = result
  TraversalCertificate.Certificates{certificate,output_certificate} = certificates
  (Storage.Buffer{reader,length,certificate},Storage.Buffer{output,output_length,output_certificate})

# Elements 0 to count - 1 of output, each combined with a sample read from
# the source; the source comes back.
def updated_into(~R: Data,~E: Data,~S: Data,~Context: Data,~read: Array<R> -> Context -> U32 -> Array<R> & S,
  ~read_preserves: @-reader: Array<R> -> @context: Context -> @index: U32 -> TraversalWitness.array_evidence(~R,reader) ->
    TraversalWitness.Sample<Array<R>,S,reader,read(reader,context,index)>,
  ~combine: E -> S -> E,~destination: Context -> U32 -> U32,+count: Nat,+context: Context,source: Storage.Buffer<R>,output: Storage.Buffer<E>) ->
  Storage.Buffer<R> & Storage.Buffer<E>:
  Storage.Buffer{reader,+length,certificate} = source
  Certificate.Certificate{+depth,+equation} = certificate
  Storage.Buffer{storage,+output_length,output_certificate} = output
  +run = {Traversal.Steps{count,0,context} : Traversal.Steps<Context>}
  read_buffers(~R,~E,length,output_length,
    Traversal.updated(~Array<R>,~E,~S,~Context,~read,~combine,~destination,run,reader,storage),
    TraversalCertificate.updated_certificates(~Array<R>,~E,~S,~Context,~TraversalWitness.array_evidence(~R),~Certificate.family(~R),~read,~combine,~destination,
      ~read_preserves,~TraversalCertificate.array_observed(~R),run,reader,storage,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},
      Certificate.Certificate{depth,equation},output_certificate))

# Elements 0 to count - 1 of output, each read from the source; the source
# comes back.
def read_into(~R: Data,~E: Data,~Context: Data,~read: Array<R> -> Context -> U32 -> Array<R> & E,
  ~read_preserves: @-reader: Array<R> -> @context: Context -> @index: U32 -> TraversalWitness.array_evidence(~R,reader) ->
    TraversalWitness.Sample<Array<R>,E,reader,read(reader,context,index)>,
  ~destination: Context -> U32 -> U32,+count: Nat,+context: Context,source: Storage.Buffer<R>,output: Storage.Buffer<E>) -> Storage.Buffer<R> & Storage.Buffer<E>:
  updated_into(~R,~E,~E,~Context,~read,~read_preserves,~Gemm.replaced(~E),~destination,count,context,source,output)

# The pairs of the lane tiles of channels channels over inner, into pairs of
# length pairs_of(channels) * inner.
def point_pairs(+channels: U32,+inner: U32,weights: Storage.Buffer<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile16>:
  read_into(~Tile.Tile8,~Tile.Tile16,~Gemm.PairPlan,~Gemm.pair_read,~GemmCertificate.pair_read,~Gemm.pair_placed,U32.to_nat((Gemm.pairs_of(channels) * inner : U32)),
    Gemm.PairPlan{Gemm.blocks(channels),inner},weights,pairs)

def panel_pairs(+channels: U32,+inner: U32,+base: U32,weights: Storage.Buffer<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile16>:
  read_into(~Tile.Tile8,~Tile.Tile16,~Gemm.PanelPlan,~Gemm.panel_pair_read,~GemmCertificate.panel_pair_read,~Gemm.panel_pair_placed,
    U32.to_nat((Gemm.pairs_of(channels) * inner : U32)),Gemm.PanelPlan{Gemm.blocks(channels),inner,base},weights,pairs)

# The count cells of the staged pairs of channels channels, mapped.
def staged_cells(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+channels: U32,+count: U32,+context: Context,staged: Storage.Buffer<Tile.Tile16>,
  output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>:
  read_into(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedPlan<Context>,~Gemm.staged_read(~Context,~map),~GemmCertificate.staged_read(~Context,~map),~Gemm.staged_placed(~Context),
    U32.to_nat(count),Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},staged,output)

def staged_slice(+slice: Gemm.StagedSlice,+count: U32,staged: Storage.Buffer<Tile.Tile16>,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>:
  read_into(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedSlice,~Gemm.staged_slice_read,~GemmCertificate.staged_slice_read,~Gemm.staged_slice_destination,U32.to_nat(count),slice,staged,output)
