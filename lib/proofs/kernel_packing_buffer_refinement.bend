# The shared stencil packer retains actual Buffer owners and exposes its tree model.
import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_buffer.bend as PackingBuffer
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ./kernel_spatial_packing_refinement.bend as Model
import ./kernel_packing_certificate.bend as PackingCertificate
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate

type Owners is Data:
  Owners{input: Buffer.Model<F32>,output: Buffer.Model<Tile.Tile8>}

def modeled(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Owners:
  (input,output) = result
  Owners{Buffer.remodel(~F32,input),Buffer.remodel(~Tile.Tile8,output)}

def represent(owners: Owners) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Owners{input,output} = owners
  (Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output))

def modeled_matches(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {represent(modeled(result)) == result : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}:
  (input,output) = result
  %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,Buffer.remodel(~F32,input)),input,Buffer.remodeled(~F32,input)) : {(_,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,output))) == result : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,output)),output,Buffer.remodeled(~Tile.Tile8,output)) : {(input,_) == result : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}
  {==}

def window(+stencil: Packing.Stencil,+count: Nat,+start: U32,input: Buffer.Model<F32>,output: Buffer.Model<Tile.Tile8>) -> Owners:
  modeled(PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output)))

def window_matches(+stencil: Packing.Stencil,+count: Nat,+start: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<Tile.Tile8>) ->
  {represent(window(stencil,count,start,input,output)) == PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output)) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}:
  modeled_matches(PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output)))

def input(owners: Owners) -> Buffer.Model<F32>:
  Owners{input,output} = owners
  input

def output(owners: Owners) -> Buffer.Model<Tile.Tile8>:
  Owners{input,output} = owners
  output

def contents(owners: Owners) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  Owners{input,output} = owners
  Shape.Both{Buffer.content(~F32,input),Buffer.content(~Tile.Tile8,output)}

def native_contents(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  (input,output) = result
  Shape.Both{Buffer.contents(~F32,input),Buffer.contents(~Tile.Tile8,output)}

def modeled_contents(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {contents(modeled(result)) == native_contents(result) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  (input,output) = result
  %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,Buffer.remodel(~F32,input)),Buffer.contents(~F32,input),Buffer.remodel_content(~F32,input)) : {Shape.Both{_,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,output))} == native_contents(result) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,output)),Buffer.contents(~Tile.Tile8,output),Buffer.remodel_content(~Tile.Tile8,output)) : {Shape.Both{Buffer.contents(~F32,input),_} == native_contents(result) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  {==}

def reflected(input_length: Nat,output_length: Nat,result: Array<F32> & Array<Tile.Tile8>) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  (input,output) = result
  Shape.Both{Buffer.Content{Tree.reflect(F32,input),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,output),output_length}}

def wrapped_contents(+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  +certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {native_contents(PackingBuffer.buffers(input_length,output_length,result,certificates)) == reflected(input_length,output_length,result) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def described(input_length: Nat,output_length: Nat,packed: Model.Packed) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  Model.Packed{input,output} = packed
  Shape.Both{Buffer.Content{input,input_length},Buffer.Content{output,output_length}}

def reflected_packed(+input_length: Nat,+output_length: Nat,+packed: Model.Packed) ->
  {reflected(input_length,output_length,Model.pack_pair(packed)) == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  Model.Packed{+input,+output} = packed
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,input)),input,StorageLaws.reflect_packed(F32,input)) :
    {Shape.Both{Buffer.Content{_,input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length}} == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output,StorageLaws.reflect_packed(Tile.Tile8,output)) :
    {Shape.Both{Buffer.Content{input,input_length},Buffer.Content{_,output_length}} == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  {==}

def window_contents(+stencil: Packing.Stencil,+count: Nat,+start: U32,+input: Buffer.Model<F32>,+output: Buffer.Model<Tile.Tile8>) ->
  {contents(window(stencil,count,start,input,output)) ==
    described(Buffer.length(~F32,input),Buffer.length(~Tile.Tile8,output),Model.packed_window(stencil,count,start,Buffer.tree(~F32,input),Buffer.tree(~Tile.Tile8,output))) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  +packed = Model.packed_window(stencil,count,start,input_tree,output_tree)
  %Equal.sym(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,contents(window(stencil,count,start,input,output)),native_contents(PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output))),
    modeled_contents(PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output)))) :
    {_ == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,native_contents(PackingBuffer.run_window(stencil,count,start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,output))),
    reflected(input_length,output_length,Packing.pack_window(stencil,count,start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,output_tree))),
    wrapped_contents(input_length,output_length,Packing.pack_window(stencil,count,start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,output_tree)),
      PackingCertificate.window_certificates(stencil,count,start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,output_tree),input_certificate,output_certificate))) :
    {_ == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Packing.pack_window(stencil,count,start,Tree.pack(F32,input_tree),Tree.pack(Tile.Tile8,output_tree)),Model.pack_pair(packed),Model.packed_window_matches(stencil,count,start,input_tree,output_tree)) :
    {reflected(input_length,output_length,_) == described(input_length,output_length,packed) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  reflected_packed(input_length,output_length,packed)

def input_visible(+owners: Owners) ->
  {Buffer.content_tree(~F32,Shape.first(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,contents(owners))) == Buffer.tree(~F32,input(owners)) : Tree.Tree<F32>}:
  Owners{input,output} = owners
  Buffer.content_tree_of(~F32,input)

def output_visible(+owners: Owners) ->
  {Buffer.content_tree(~Tile.Tile8,Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,contents(owners))) == Buffer.tree(~Tile.Tile8,output(owners)) : Tree.Tree<Tile.Tile8>}:
  Owners{input,output} = owners
  Buffer.content_tree_of(~Tile.Tile8,output)

def described_input(+input_length: Nat,+output_length: Nat,+packed: Model.Packed) ->
  {Buffer.content_tree(~F32,Shape.first(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,described(input_length,output_length,packed))) == Model.packed_input(packed) : Tree.Tree<F32>}:
  Model.Packed{input,output} = packed
  {==}

def window_input_tree(+stencil: Packing.Stencil,+count: Nat,+start: U32,+source: Buffer.Model<F32>,+workspace: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~F32,input(window(stencil,count,start,source,workspace))) == Buffer.tree(~F32,source) : Tree.Tree<F32>}:
  +owners = window(stencil,count,start,source,workspace)
  +packed = Model.packed_window(stencil,count,start,Buffer.tree(~F32,source),Buffer.tree(~Tile.Tile8,workspace))
  %input_visible(owners) : {_ == Buffer.tree(~F32,source) : Tree.Tree<F32>}
  %Equal.sym(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,contents(owners),described(Buffer.length(~F32,source),Buffer.length(~Tile.Tile8,workspace),packed),window_contents(stencil,count,start,source,workspace)) :
    {Buffer.content_tree(~F32,Shape.first(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,_)) == Buffer.tree(~F32,source) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<F32>,Buffer.content_tree(~F32,Shape.first(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,described(Buffer.length(~F32,source),Buffer.length(~Tile.Tile8,workspace),packed))),Model.packed_input(packed),described_input(Buffer.length(~F32,source),Buffer.length(~Tile.Tile8,workspace),packed)) :
    {_ == Buffer.tree(~F32,source) : Tree.Tree<F32>}
  Model.packed_window_input(stencil,count,start,Buffer.tree(~F32,source),Buffer.tree(~Tile.Tile8,workspace))

def described_output(+input_length: Nat,+output_length: Nat,+packed: Model.Packed) ->
  {Buffer.content_tree(~Tile.Tile8,Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,described(input_length,output_length,packed))) == Model.packed_output(packed) : Tree.Tree<Tile.Tile8>}:
  Model.Packed{input,output} = packed
  {==}

def window_output_tree(+stencil: Packing.Stencil,+count: Nat,+start: U32,+source: Buffer.Model<F32>,+workspace: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,output(window(stencil,count,start,source,workspace))) == Model.packed_output(Model.packed_window(stencil,count,start,Buffer.tree(~F32,source),Buffer.tree(~Tile.Tile8,workspace))) : Tree.Tree<Tile.Tile8>}:
  +owners = window(stencil,count,start,source,workspace)
  +packed = Model.packed_window(stencil,count,start,Buffer.tree(~F32,source),Buffer.tree(~Tile.Tile8,workspace))
  %output_visible(owners) : {_ == Model.packed_output(packed) : Tree.Tree<Tile.Tile8>}
  %Equal.sym(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,contents(owners),described(Buffer.length(~F32,source),Buffer.length(~Tile.Tile8,workspace),packed),window_contents(stencil,count,start,source,workspace)) :
    {Buffer.content_tree(~Tile.Tile8,Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,_)) == Model.packed_output(packed) : Tree.Tree<Tile.Tile8>}
  described_output(Buffer.length(~F32,source),Buffer.length(~Tile.Tile8,workspace),packed)
