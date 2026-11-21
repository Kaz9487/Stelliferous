# The certified GEMM boundary preserves source contents and exposes its tree model.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as Native
import ../kernel_tile.bend as Tile
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes

type Contents<-E: Data> is Data:
  Contents{row_tiles: Buffer.Content<Tile.Tile8>,lane_tiles: Buffer.Content<Tile.Tile8>,bias: Buffer.Content<F32>,output: Buffer.Content<E>}

def contents(~E: Data,state: Native.Kept<E>) -> Contents<E>:
  Native.Kept{row_tiles,lane_tiles,bias,output} = state
  Contents{Buffer.contents(~Tile.Tile8,row_tiles),Buffer.contents(~Tile.Tile8,lane_tiles),Buffer.contents(~F32,bias),Buffer.contents(~E,output)}

def operand_contents(~E: Data,+row_length: Nat,+lane_length: Nat,+bias_length: Nat,+length: Nat,operands: Gemm.Operands<E,Array<F32>>) -> Contents<E>:
  Gemm.Operands{row_tiles,lane_tiles,bias,output} = operands
  Contents{Buffer.Content{Tree.reflect(Tile.Tile8,row_tiles),row_length},Buffer.Content{Tree.reflect(Tile.Tile8,lane_tiles),lane_length},
    Buffer.Content{Tree.reflect(F32,bias),bias_length},Buffer.Content{Tree.reflect(E,output),length}}

def kept_contents(~E: Data,+row_length: Nat,+lane_length: Nat,+bias_length: Nat,+length: Nat,operands: Gemm.Operands<E,Array<F32>>,
  certificates: GemmCertificate.Certified<E,operands>) ->
  {contents(~E,Native.kept(~E,operands,row_length,lane_length,bias_length,length,certificates)) ==
    operand_contents(~E,row_length,lane_length,bias_length,length,operands) : Contents<E>}:
  Gemm.Operands{row_tiles,lane_tiles,bias,output} = operands
  GemmCertificate.Certified{row_certificate,lane_certificate,bias_certificate,certificate} = certificates
  {==}

def model_contents(~E: Data,+row_length: Nat,+lane_length: Nat,+bias_length: Nat,+length: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +bias: Tree.Tree<F32>,+output: Writes.Program<E>) ->
  {operand_contents(~E,row_length,lane_length,bias_length,length,GemmLaws.represent(~E,GemmLaws.Operands{row_tiles,lane_tiles,bias,output})) ==
    Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{bias,bias_length},Buffer.Content{Writes.evaluate(E,output),length}} : Contents<E>}:
  +result = Writes.evaluate(E,output)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,lane_tiles)),lane_tiles,StorageLaws.reflect_packed(Tile.Tile8,lane_tiles)) :
    {Contents{Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,row_tiles)),row_length},Buffer.Content{_,lane_length},
        Buffer.Content{Tree.reflect(F32,Tree.pack(F32,bias)),bias_length},Buffer.Content{Tree.reflect(E,Tree.pack(E,result)),length}} ==
      Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{bias,bias_length},Buffer.Content{result,length}} : Contents<E>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,row_tiles)),row_tiles,StorageLaws.reflect_packed(Tile.Tile8,row_tiles)) :
    {Contents{Buffer.Content{_,row_length},Buffer.Content{lane_tiles,lane_length},
        Buffer.Content{Tree.reflect(F32,Tree.pack(F32,bias)),bias_length},Buffer.Content{Tree.reflect(E,Tree.pack(E,result)),length}} ==
      Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{bias,bias_length},Buffer.Content{result,length}} : Contents<E>}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,bias)),bias,StorageLaws.reflect_packed(F32,bias)) :
    {Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{_,bias_length},Buffer.Content{Tree.reflect(E,Tree.pack(E,result)),length}} ==
      Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{bias,bias_length},Buffer.Content{result,length}} : Contents<E>}
  Equal.cong(Tree.Tree<E>,Contents<E>,tree => Contents{Buffer.Content{row_tiles,row_length},Buffer.Content{lane_tiles,lane_length},Buffer.Content{bias,bias_length},Buffer.Content{tree,length}},
    Tree.reflect(E,Tree.pack(E,result)),result,StorageLaws.reflect_packed(E,result))

# A run of a plan keeps its sources and stores the model run's tree.
def plan_contents(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: GemmLaws.Order,~ordered: GemmLaws.Ordered(~times,order),~writer: GemmLaws.Writer,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<GemmLaws.Element(writer),Array<F32>> -> Gemm.Written<GemmLaws.Element(writer),Array<F32>>,
  ~keeps: GemmCertificate.Keeps(~GemmLaws.Element(writer),~write),~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~stored: GemmLaws.Stored(~writer,~write,~value,~close),+count: Nat,+plan: Plan,
  +row_tiles: Buffer.Model<Tile.Tile8>,+lane_tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<GemmLaws.Element(writer)>) ->
  {contents(~GemmLaws.Element(writer),Native.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,~keeps,count,plan,
      Buffer.pack(~Tile.Tile8,row_tiles),Buffer.pack(~Tile.Tile8,lane_tiles),Buffer.pack(~F32,bias),Buffer.pack(~GemmLaws.Element(writer),output))) ==
    Contents{Buffer.content(~Tile.Tile8,row_tiles),Buffer.content(~Tile.Tile8,lane_tiles),Buffer.content(~F32,bias),
      Buffer.Content{GemmLaws.output_tree(~GemmLaws.Element(writer),GemmLaws.run(~Plan,~Cursor,~writer,~job_at,~advance,~order,
        ~value,~close,count,plan,GemmLaws.Operands{Buffer.tree(~Tile.Tile8,row_tiles),Buffer.tree(~Tile.Tile8,lane_tiles),Buffer.tree(~F32,bias),
          Writes.Initial{Buffer.tree(~GemmLaws.Element(writer),output)}})),Buffer.length(~GemmLaws.Element(writer),output)}} : Contents<GemmLaws.Element(writer)>}:
  Buffer.Model{+wt,+wl,+wc} = row_tiles
  Buffer.Model{+it,+il,+ic} = lane_tiles
  Buffer.Model{+bt,+bl,+bc} = bias
  Buffer.Model{+ot,+ol,+oc} = output
  +operands = {GemmLaws.Operands{wt,it,bt,Writes.Initial{ot}} : GemmLaws.Operands<GemmLaws.Element(writer)>}
  +model = GemmLaws.run(~Plan,~Cursor,~writer,~job_at,~advance,~order,~value,~close,count,plan,operands)
  native = Gemm.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,count,plan,
    GemmLaws.represent(~GemmLaws.Element(writer),operands))
  -result = {Contents{Buffer.Content{wt,wl},Buffer.Content{it,il},Buffer.Content{bt,bl},Buffer.Content{GemmLaws.output_tree(~GemmLaws.Element(writer),model),ol}} :
    Contents<GemmLaws.Element(writer)>}
  certificates = GemmCertificate.certificates(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,~keeps,count,plan,
    Tree.pack(Tile.Tile8,wt),Tree.pack(Tile.Tile8,it),Tree.pack(F32,bt),Tree.pack(GemmLaws.Element(writer),ot),wc,ic,bc,oc)
  %Equal.sym(Contents<GemmLaws.Element(writer)>,contents(~GemmLaws.Element(writer),Native.kept(~GemmLaws.Element(writer),native,wl,il,bl,ol,certificates)),
    operand_contents(~GemmLaws.Element(writer),wl,il,bl,ol,native),kept_contents(~GemmLaws.Element(writer),wl,il,bl,ol,native,certificates)) :
    {_ == result : Contents<GemmLaws.Element(writer)>}
  %Equal.sym(Gemm.Operands<GemmLaws.Element(writer),Array<F32>>,native,GemmLaws.represent(~GemmLaws.Element(writer),model),
    GemmLaws.run_matches(~Plan,~Cursor,~writer,~job_at,~advance,~times,~order,~ordered,
      ~write,~value,~close,~stored,count,plan,operands)) :
    {operand_contents(~GemmLaws.Element(writer),wl,il,bl,ol,_) == result : Contents<GemmLaws.Element(writer)>}
  %Equal.sym(GemmLaws.Operands<GemmLaws.Element(writer)>,model,GemmLaws.Operands{wt,it,bt,GemmLaws.operand_output(~GemmLaws.Element(writer),model)},
    GemmJobs.run_keeps(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,wt,it,bt,Writes.Initial{ot})) :
    {operand_contents(~GemmLaws.Element(writer),wl,il,bl,ol,GemmLaws.represent(~GemmLaws.Element(writer),_)) == result : Contents<GemmLaws.Element(writer)>}
  model_contents(~GemmLaws.Element(writer),wl,il,bl,ol,wt,it,bt,GemmLaws.operand_output(~GemmLaws.Element(writer),model))

# A run of a plan keeps its sources and applies the writes of its jobs
# to its output.
def plan_writes(~Plan: Data,~Cursor: Data,~job_at: Plan -> U32 -> Gemm.Job<Cursor>,~advance: Cursor -> Gemm.Step<Cursor>,~times: F32 -> F32 -> F32,~order: GemmLaws.Order,~ordered: GemmLaws.Ordered(~times,order),~writer: GemmLaws.Writer,
  ~write: @+bias_start: U32 -> @+store: Gemm.Store -> @+accumulators: Gemm.Accumulators -> Gemm.Written<GemmLaws.Element(writer),Array<F32>> -> Gemm.Written<GemmLaws.Element(writer),Array<F32>>,
  ~keeps: GemmCertificate.Keeps(~GemmLaws.Element(writer),~write),~value: Tree.Tree<F32> -> U32 -> F32,~close: F32 -> F32 -> F32,
  ~stored: GemmLaws.Stored(~writer,~write,~value,~close),+count: Nat,+plan: Plan,
  +row_tiles: Buffer.Model<Tile.Tile8>,+lane_tiles: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<GemmLaws.Element(writer)>) ->
  {contents(~GemmLaws.Element(writer),Native.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,~keeps,count,plan,
      Buffer.pack(~Tile.Tile8,row_tiles),Buffer.pack(~Tile.Tile8,lane_tiles),Buffer.pack(~F32,bias),Buffer.pack(~GemmLaws.Element(writer),output))) ==
    Contents{Buffer.content(~Tile.Tile8,row_tiles),Buffer.content(~Tile.Tile8,lane_tiles),Buffer.content(~F32,bias),
      Buffer.Content{Writes.pure(GemmLaws.Element(writer),GemmJobs.all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,
        GemmJobs.Readers{Buffer.tree(~Tile.Tile8,row_tiles),Buffer.tree(~Tile.Tile8,lane_tiles),Buffer.tree(~F32,bias)}),Buffer.tree(~GemmLaws.Element(writer),output)),
        Buffer.length(~GemmLaws.Element(writer),output)}} : Contents<GemmLaws.Element(writer)>}:
  +operands = {GemmLaws.Operands{Buffer.tree(~Tile.Tile8,row_tiles),Buffer.tree(~Tile.Tile8,lane_tiles),Buffer.tree(~F32,bias),Writes.Initial{Buffer.tree(~GemmLaws.Element(writer),output)}} :
    GemmLaws.Operands<GemmLaws.Element(writer)>}
  +written = GemmLaws.output_tree(~GemmLaws.Element(writer),GemmLaws.run(~Plan,~Cursor,~writer,~job_at,~advance,~order,
    ~value,~close,count,plan,operands))
  +wanted = Writes.pure(GemmLaws.Element(writer),GemmJobs.all_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,
    GemmJobs.readers_of(~GemmLaws.Element(writer),operands)),Buffer.tree(~GemmLaws.Element(writer),output))
  +length = Buffer.length(~GemmLaws.Element(writer),output)
  Equal.trans(Contents<GemmLaws.Element(writer)>,contents(~GemmLaws.Element(writer),Native.run(~Plan,~Cursor,~GemmLaws.Element(writer),~job_at,~advance,~times,~write,~keeps,count,plan,
      Buffer.pack(~Tile.Tile8,row_tiles),Buffer.pack(~Tile.Tile8,lane_tiles),Buffer.pack(~F32,bias),Buffer.pack(~GemmLaws.Element(writer),output))),
    Contents{Buffer.content(~Tile.Tile8,row_tiles),Buffer.content(~Tile.Tile8,lane_tiles),Buffer.content(~F32,bias),Buffer.Content{written,length}},
    Contents{Buffer.content(~Tile.Tile8,row_tiles),Buffer.content(~Tile.Tile8,lane_tiles),Buffer.content(~F32,bias),Buffer.Content{wanted,length}},
    plan_contents(~Plan,~Cursor,~job_at,~advance,~times,~order,~ordered,~writer,~write,~keeps,~value,~close,~stored,count,plan,row_tiles,lane_tiles,bias,output),
    Equal.cong(Tree.Tree<GemmLaws.Element(writer)>,Contents<GemmLaws.Element(writer)>,
      tree => Contents{Buffer.content(~Tile.Tile8,row_tiles),Buffer.content(~Tile.Tile8,lane_tiles),Buffer.content(~F32,bias),Buffer.Content{tree,length}},
      written,wanted,GemmJobs.run_writes(~Plan,~Cursor,~job_at,~advance,~order,~writer,~value,~close,count,0,plan,operands)))
