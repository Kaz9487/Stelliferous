# The certified GEMM boundary preserves source contents and exposes its tree model.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as Native
import ../kernel_tile.bend as Tile
import ./kernel_gemm_refinement.bend as Model
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate

type Contents is Data:
  Contents{input: Buffer.Content<Tile.Tile8>,weights: Buffer.Content<Tile.Tile8>,bias: Buffer.Content<F32>,output: Buffer.Content<F32>}

def contents(state: Native.Kept) -> Contents:
  Native.Kept{input,weights,bias,output} = state
  Contents{Buffer.contents(~Tile.Tile8,input),Buffer.contents(~Tile.Tile8,weights),Buffer.contents(~F32,bias),Buffer.contents(~F32,output)}

def raw_contents(il: Nat,wl: Nat,bl: Nat,ol: Nat,state: Gemm.Buffers) -> Contents:
  Gemm.Buffers{input,weights,bias,output} = state
  Contents{Buffer.Content{Tree.reflect(Tile.Tile8,input),il},Buffer.Content{Tree.reflect(Tile.Tile8,weights),wl},
    Buffer.Content{Tree.reflect(F32,bias),bl},Buffer.Content{Tree.reflect(F32,output),ol}}

def kept_contents(+il: Nat,+wl: Nat,+bl: Nat,+ol: Nat,state: Gemm.Buffers,certificates: GemmCertificate.Kept<state>) ->
  {contents(Native.kept(state,il,wl,bl,ol,certificates)) == raw_contents(il,wl,bl,ol,state) : Contents}:
  Gemm.Buffers{input,weights,bias,output} = state
  GemmCertificate.Kept{ic,wc,bc,oc} = certificates
  {==}

def raw_kept(+rows: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) ->
  {Gemm.run_keeping(~True{},rows,Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,input)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights)),Tree.pack(F32,Buffer.tree(~F32,bias)),Tree.pack(F32,Buffer.tree(~F32,output)),columns,inner,start,output_columns) ==
    Gemm.Buffers{Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,input)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights)),Tree.pack(F32,Buffer.tree(~F32,bias)),
      Gemm.run_at(~True{},rows,Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,input)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights)),Tree.pack(F32,Buffer.tree(~F32,bias)),Tree.pack(F32,Buffer.tree(~F32,output)),columns,inner,start,output_columns)} : Gemm.Buffers}:
  Buffer.Model{+it,il,+ic} = input
  Buffer.Model{+wt,wl,+wc} = weights
  Buffer.Model{+bt,bl,+bc} = bias
  Buffer.Model{+ot,ol,Certificate.Certificate{+depth,+equation}} = output
  GemmCertificate.kept_equation(depth,rows,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Tree.pack(F32,ot),columns,inner,start,output_columns,
    Certificate.witness(Tile.Tile8,Tree.pack(Tile.Tile8,it),ic),Certificate.witness(Tile.Tile8,Tree.pack(Tile.Tile8,wt),wc),Certificate.witness(F32,Tree.pack(F32,bt),bc),
    Certificate.witness_at(F32,depth,Tree.pack(F32,ot),equation))

def raw_packed(+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,+result: Tree.Tree<F32>) ->
  {raw_contents(Buffer.length(~Tile.Tile8,input),Buffer.length(~Tile.Tile8,weights),Buffer.length(~F32,bias),Buffer.length(~F32,output),
      Gemm.Buffers{Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,input)),Tree.pack(Tile.Tile8,Buffer.tree(~Tile.Tile8,weights)),Tree.pack(F32,Buffer.tree(~F32,bias)),Tree.pack(F32,result)}) ==
    Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{result,Buffer.length(~F32,output)}} : Contents}:
  Buffer.Model{+it,il,ic} = input
  Buffer.Model{+wt,wl,wc} = weights
  Buffer.Model{+bt,bl,bc} = bias
  Buffer.Model{ot,+ol,oc} = output
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,it)),it,StorageLaws.reflect_packed(Tile.Tile8,it)) :
    {Contents{Buffer.Content{_,il},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,wt)),wl},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,bt)),bl},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,result)),ol}} == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{result,ol}} : Contents}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,wt)),wt,StorageLaws.reflect_packed(Tile.Tile8,wt)) :
    {Contents{Buffer.Content{it,il},Buffer.Content{_,wl},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,bt)),bl},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,result)),ol}} == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{result,ol}} : Contents}
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,bt)),bt,StorageLaws.reflect_packed(F32,bt)) :
    {Contents{Buffer.Content{it,il},Buffer.Content{wt,wl},Buffer.Content{_,bl},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,result)),ol}} == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{result,ol}} : Contents}
  Equal.cong(Tree.Tree<F32>,Contents,tree => Contents{Buffer.Content{it,il},Buffer.Content{wt,wl},Buffer.Content{bt,bl},Buffer.Content{tree,ol}},
    Tree.reflect(F32,Tree.pack(F32,result)),result,StorageLaws.reflect_packed(F32,result))

def run_contents(+rows: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>,+output: Buffer.Model<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) ->
  {contents(Native.run_keeping(rows,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Buffer.pack(~F32,output),columns,inner,start,output_columns)) ==
    Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),
      Buffer.Content{Model.run_at(~True{},rows,Buffer.tree(~Tile.Tile8,input),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias),Buffer.tree(~F32,output),columns,inner,start,output_columns),Buffer.length(~F32,output)}} : Contents}:
  Buffer.Model{+it,+il,+ic} = input
  Buffer.Model{+wt,+wl,+wc} = weights
  Buffer.Model{+bt,+bl,+bc} = bias
  Buffer.Model{+ot,+ol,+oc} = output
  state = Gemm.run_keeping(~True{},rows,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Tree.pack(F32,ot),columns,inner,start,output_columns)
  +certificates = GemmCertificate.kept(rows,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Tree.pack(F32,ot),columns,inner,start,output_columns,ic,wc,bc,oc)
  %Equal.sym(Contents,contents(Native.kept(state,il,wl,bl,ol,certificates)),raw_contents(il,wl,bl,ol,state),kept_contents(il,wl,bl,ol,state,certificates)) :
    {_ == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{Model.run_at(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns),ol}} : Contents}
  %Equal.sym(Gemm.Buffers,state,Gemm.Buffers{Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Gemm.run_at(~True{},rows,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Tree.pack(F32,ot),columns,inner,start,output_columns)},
    raw_kept(rows,input,weights,bias,output,columns,inner,start,output_columns)) :
    {raw_contents(il,wl,bl,ol,_) == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{Model.run_at(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns),ol}} : Contents}
  %Equal.sym(Array<F32>,Gemm.run_at(~True{},rows,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),Tree.pack(F32,ot),columns,inner,start,output_columns),
    Tree.pack(F32,Model.run_at(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns)),Model.run_at_matches(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns)) :
    {raw_contents(il,wl,bl,ol,Gemm.Buffers{Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile8,wt),Tree.pack(F32,bt),_}) == Contents{Buffer.content(~Tile.Tile8,input),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias),Buffer.Content{Model.run_at(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns),ol}} : Contents}
  raw_packed(input,weights,bias,output,Model.run_at(~True{},rows,it,wt,bt,ot,columns,inner,start,output_columns))
