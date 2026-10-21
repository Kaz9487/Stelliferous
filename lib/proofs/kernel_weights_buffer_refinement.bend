# The Buffer wrapper preserves the existing weight-reorder content theorem.
# No reduction, GEMM loop, or numerical accumulation is opened here.
import Base
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ./storage_shape.bend as Shape
import ./storage_certificate.bend as Certificate
import ./kernel_weights_certificate.bend as WeightCertificate
import ./kernel_weights_refinement.bend as WeightsLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./kernel_pipeline_buffer_shape.bend as Footprint
import ./tensor_allocation_depth.bend as Depth

def reordered(+rows: U32,+inner: U32,+weights: Buffer.Model<F32>) ->
  {Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))) ==
    Buffer.Content{WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)),U32.to_nat((((rows + 7) / 8) * inner : U32))} : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,length,certificate} = weights
  +count = U32.to_nat((((rows + 7) / 8) * inner : U32))
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,tree)),Tree.pack(Tile.Tile8,WeightsLaws.run(rows,inner,tree)),WeightsLaws.run_matches(rows,inner,tree)) :
    {Buffer.Content{Tree.reflect(Tile.Tile8,_),count} == Buffer.Content{WeightsLaws.run(rows,inner,tree),count} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,WeightsLaws.run(rows,inner,tree))),WeightsLaws.run(rows,inner,tree),StorageLaws.reflect_packed(Tile.Tile8,WeightsLaws.run(rows,inner,tree))) :
    {Buffer.Content{_,count} == Buffer.Content{WeightsLaws.run(rows,inner,tree),count} : Buffer.Content<Tile.Tile8>}
  {==}

def mapped_contents(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  (source,packed) = result
  Shape.Both{Buffer.contents(~F32,source),Buffer.contents(~Tile.Tile8,packed)}

def mapped_reflected(+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  (source,packed) = result
  Shape.Both{Buffer.Content{Tree.reflect(F32,source),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,packed),output_length}}

def mapped_wrapped(+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: WeightCertificate.MappedCertificates<result>) ->
  {mapped_contents(GemmBuffer.mapped_buffers(input_length,output_length,result,certificates)) == mapped_reflected(input_length,output_length,result) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  match result certificates:
    case (source,packed) WeightCertificate.MappedCertificates{source_certificate,packed_certificate}: {==}

def mapped_description(+input_length: Nat,+output_length: Nat,owners: WeightsLaws.Owners) -> Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>:
  WeightsLaws.Owners{source,packed} = owners
  Shape.Both{Buffer.Content{source,input_length},Buffer.Content{packed,output_length}}

def mapped_packed(+input_length: Nat,+output_length: Nat,+owners: WeightsLaws.Owners) ->
  {mapped_reflected(input_length,output_length,WeightsLaws.represent(owners)) == mapped_description(input_length,output_length,owners) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  WeightsLaws.Owners{+source,+packed} = owners
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,source)),source,StorageLaws.reflect_packed(F32,source)) : {Shape.Both{Buffer.Content{_,input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,packed)),output_length}} == mapped_description(input_length,output_length,owners) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,packed)),packed,StorageLaws.reflect_packed(Tile.Tile8,packed)) : {Shape.Both{Buffer.Content{source,input_length},Buffer.Content{_,output_length}} == mapped_description(input_length,output_length,owners) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  {==}

def reordered_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {mapped_contents(GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights))) ==
    mapped_description(Buffer.length(~F32,weights),U32.to_nat((((rows + 7) / 8) * inner : U32)),WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}:
  Buffer.Model{+tree,+length,+certificate} = weights
  +count = U32.to_nat((((rows + 7) / 8) * inner : U32))
  +owners = WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,tree)
  -result = Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree))
  %Equal.sym(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,mapped_contents(GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights))),mapped_reflected(length,count,result),
    mapped_wrapped(length,count,Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree)),WeightCertificate.certificates_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree),certificate))) :
    {_ == mapped_description(length,count,owners) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,result,WeightsLaws.represent(owners),WeightsLaws.run_mapped_matches_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,tree)) :
    {mapped_reflected(length,count,_) == mapped_description(length,count,owners) : Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>}
  mapped_packed(length,count,owners)

def source_wrapped(+length: Nat,+count: Nat,-input: Array<F32>,result: Array<F32> & Array<Tile.Tile8>,+depth: Nat,
  +equation: {input == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,input))) : Array<F32>},
  output: Certificate.Certificate<Tile.Tile8,Weights.reordered(result)>,same: {input == Weights.source(result) : Array<F32>}) ->
  {Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,GemmBuffer.mapped_buffers(length,count,result,
    WeightCertificate.MappedCertificates{WeightCertificate.retained(input,Weights.source(result),depth,equation,same),output})) == Storage.Buffer{input,length,Certificate.Certificate{depth,equation}} : Storage.Buffer<F32>}:
  (reader,packed) = result
  %proof@same : {Storage.Buffer{_,length,WeightCertificate.retained(input,_,depth,equation,proof)} == Storage.Buffer{input,length,Certificate.Certificate{depth,equation}} : Storage.Buffer<F32>}
  {==}

def source_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights))) == Buffer.pack(~F32,weights) : Storage.Buffer<F32>}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = weights
  source_wrapped(length,U32.to_nat((((rows + 7) / 8) * inner : U32)),Tree.pack(F32,tree),Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree)),depth,equation,
    Certificate.Certificate{Tensors.depth_for((((rows + 7) / 8) * inner : U32)),WeightCertificate.equation_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree),depth,equation)},
    WeightCertificate.input_equation_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Tree.pack(F32,tree),depth,equation))

# Observe the returned packed owner; its actual certificate is retained.

def source_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,GemmBuffer.reorder_mapped(~Context,~offset,rows,inner,row_stride,context,Buffer.pack(~F32,weights))) == Buffer.pack(~F32,weights) : Storage.Buffer<F32>}:
  Buffer.Model{+tree,+length,+certificate} = weights
  source_mapped_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,weights)

def packed_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,weights: Buffer.Model<F32>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights))))

def packed_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,weights: Buffer.Model<F32>) -> Buffer.Model<Tile.Tile8>:
  Buffer.Model{+tree,+length,+certificate} = weights
  packed_mapped_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,weights)

def mapped_second(result: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,mapped_contents(result)) == Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,result)) : Buffer.Content<Tile.Tile8>}:
  (source,packed) = result
  {==}

def description_second(+input_length: Nat,+output_length: Nat,+owners: WeightsLaws.Owners) ->
  {Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,mapped_description(input_length,output_length,owners)) == Buffer.Content{WeightsLaws.output(owners),output_length} : Buffer.Content<Tile.Tile8>}:
  WeightsLaws.Owners{source,packed} = owners
  {==}

def packed_mapped_content_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {Buffer.content(~Tile.Tile8,packed_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights)) == Buffer.Content{WeightsLaws.output(WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))),U32.to_nat((((rows + 7) / 8) * inner : U32))} : Buffer.Content<Tile.Tile8>}:
  native = GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights))
  packed = Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,GemmBuffer.reorder_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.pack(~F32,weights)))
  +count = U32.to_nat((((rows + 7) / 8) * inner : U32))
  +owners = WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))
  +actual = Buffer.content(~Tile.Tile8,packed_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights))
  Equal.trans(Buffer.Content<Tile.Tile8>,actual,Buffer.contents(~Tile.Tile8,packed),Buffer.Content{WeightsLaws.output(owners),count},Buffer.remodel_content(~Tile.Tile8,packed),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,packed),Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,mapped_description(Buffer.length(~F32,weights),count,owners)),Buffer.Content{WeightsLaws.output(owners),count},
      %mapped_second(native) : {_ == Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>,mapped_description(Buffer.length(~F32,weights),count,owners)) : Buffer.Content<Tile.Tile8>}
      Equal.cong(Shape.Both<Buffer.Content<F32>,Buffer.Content<Tile.Tile8>>,Buffer.Content<Tile.Tile8>,Shape.second(Buffer.Content<F32>,Buffer.Content<Tile.Tile8>),mapped_contents(native),mapped_description(Buffer.length(~F32,weights),count,owners),reordered_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights)),description_second(Buffer.length(~F32,weights),count,owners)))

def packed_mapped_tree_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {Buffer.tree(~Tile.Tile8,packed_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights)) == WeightsLaws.output(WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))) : Tree.Tree<Tile.Tile8>}:
  +model = packed_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights)
  %Buffer.content_tree_of(~Tile.Tile8,model) : {_ == WeightsLaws.output(WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))) : Tree.Tree<Tile.Tile8>}
  Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.content(~Tile.Tile8,model),Buffer.Content{WeightsLaws.output(WeightsLaws.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,Buffer.tree(~F32,weights))),U32.to_nat((((rows + 7) / 8) * inner : U32))},packed_mapped_content_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights))

def packed_mapped_tree(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,+weights: Buffer.Model<F32>) ->
  {Buffer.tree(~Tile.Tile8,packed_mapped(~Context,~offset,rows,inner,row_stride,context,weights)) == WeightsLaws.output(WeightsLaws.run_mapped(~Context,~offset,rows,inner,row_stride,context,Buffer.tree(~F32,weights))) : Tree.Tree<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = weights
  packed_mapped_tree_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,weights)
