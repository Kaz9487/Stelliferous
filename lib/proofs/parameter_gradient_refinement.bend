# Bias numerical refinement uses the common image and group traversal; the
# outputs are written one after another into the result.
# Source represents the two trees used by the weight contract.
import Base
import ../kernel_parameter_gradient.bend as Kernel
import ../traversal_grouped.bend as Grouped
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as Native
import ./grouped_reduction_refinement.bend as Reduction
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer

type Source is Data:
  Source{input: Tree.Tree<F32>,gradient: Tree.Tree<F32>}

def bias_pack(source: Tree.Tree<F32>) -> Array<F32>:
  Tree.pack(F32,source)

def bias_sample(source: Tree.Tree<F32>,context: Kernel.Context,position: Nat) -> F32:
  Tree.read(F32,source,Kernel.bias_address(context,position))

def bias_cell(+source: Tree.Tree<F32>,+context: Kernel.Context,+position: Nat) ->
  {Kernel.bias_at(bias_pack(source),context,position) == (bias_pack(source),bias_sample(source,context,position)) : Array<F32> & F32}:
  Native.read(F32,source,Kernel.bias_address(context,position))

def image_values(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,ranges,source,Kernel.Context{work,image,element})) <>
        image_values(~Source,~sample,rest,1n+image,element,work,ranges,source)

def images(~Reader: Type,~Source: Data,~pack: Source -> Reader,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~sample: Source -> Kernel.Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Kernel.Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source) ->
  {Kernel.image_values(~Reader,~read,remaining,image,element,work,ranges,pack(source)) ==
    (pack(source),image_values(~Source,~sample,remaining,image,element,work,ranges,source)) : Reader & List<&2,F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Reader & F32,Grouped.run(~Reader,~Kernel.Context,~read,ranges,pack(source),Kernel.Context{work,image,element}),
        (pack(source),Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,ranges,source,Kernel.Context{work,image,element}))),
        Reduction.run(~Reader,~Source,~Kernel.Context,~pack,~read,~sample,~matches,ranges,source,Kernel.Context{work,image,element})) :
        {Kernel.image_tail(~Reader,_,reader => Kernel.image_values(~Reader,~read,rest,1n+image,element,work,ranges,reader)) ==
          (pack(source),image_values(~Source,~sample,1n+rest,image,element,work,ranges,source)) : Reader & List<&2,F32>}
      %Equal.sym(Reader & List<&2,F32>,Kernel.image_values(~Reader,~read,rest,1n+image,element,work,ranges,pack(source)),
        (pack(source),image_values(~Source,~sample,rest,1n+image,element,work,ranges,source)),images(~Reader,~Source,~pack,~read,~sample,~matches,rest,1n+image,element,work,ranges,source)) :
        {Grouped.prepend(~Reader,Grouped.tree(Reduction.group_values(~Source,~Kernel.Context,~sample,ranges,source,Kernel.Context{work,image,element})),_) ==
          (pack(source),image_values(~Source,~sample,1n+rest,image,element,work,ranges,source)) : Reader & List<&2,F32>}
      {==}

def element(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,+work: Kernel.Work,index: Nat,+ranges: List<&2,Grouped.Range>,source: Source) -> F32:
  Kernel.Work{images,c,o,h,w,k,s,p,rows,columns,g} = work
  Grouped.tree(image_values(~Source,~sample,images,0n,index,work,ranges,source))

def element_matches(~Reader: Type,~Source: Data,~pack: Source -> Reader,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~sample: Source -> Kernel.Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Kernel.Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  +work: Kernel.Work,+index: Nat,+ranges: List<&2,Grouped.Range>,+source: Source) ->
  {Kernel.element(~Reader,~read,work,index,ranges,pack(source)) == (pack(source),element(~Source,~sample,work,index,ranges,source)) : Reader & F32}:
  Kernel.Work{+count,c,o,h,w,k,s,p,rows,columns,g} = work
  Equal.cong(Reader & List<&2,F32>,Reader & F32,pair => Grouped.reduced(~Reader,pair),Kernel.image_values(~Reader,~read,count,0n,index,work,ranges,pack(source)),
    (pack(source),image_values(~Source,~sample,count,0n,index,work,ranges,source)),images(~Reader,~Source,~pack,~read,~sample,~matches,count,0n,index,work,ranges,source))

def values(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source) -> List<&2,F32>:
  match remaining:
    case 0n: []
    case 1n+rest: element(~Source,~sample,work,index,ranges,source) <> values(~Source,~sample,rest,1n+index,work,ranges,source)

def bias_values(+work: Kernel.Work,source: Tree.Tree<F32>) -> List<&2,F32>:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values(~Tree.Tree<F32>,~bias_sample,o,0n,work,Grouped.ranges(rows,columns,g),source)

# The write loop on packed storage is the model loop: each reduced output is
# written at its position, so the result holds the values from position on.
def written_model(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source,result: Buffer.Model<F32>) -> Buffer.Model<F32>:
  match remaining:
    case 0n: result
    case 1n+rest:
      written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,
        Buffer.remodel(~F32,Storage.write(~F32,position,element(~Source,~sample,work,index,ranges,source),Buffer.pack(~F32,result))))

def written(~Reader: Type,~Source: Data,~pack: Source -> Reader,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~sample: Source -> Kernel.Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Kernel.Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,+source: Source,+result: Buffer.Model<F32>) ->
  {Kernel.written(~Reader,~read,remaining,index,position,work,ranges,pack(source),Buffer.pack(~F32,result)) ==
    (pack(source),Buffer.pack(~F32,written_model(~Source,~sample,remaining,index,position,work,ranges,source,result))) : Reader & Storage.Buffer<F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +value = element(~Source,~sample,work,index,ranges,source)
      +next = Buffer.remodel(~F32,Storage.write(~F32,position,value,Buffer.pack(~F32,result)))
      %Equal.sym(Reader & F32,Kernel.element(~Reader,~read,work,index,ranges,pack(source)),(pack(source),value),
        element_matches(~Reader,~Source,~pack,~read,~sample,~matches,work,index,ranges,source)) :
        {Kernel.stored(~Reader,position,Buffer.pack(~F32,result),_,
            reader => result => Kernel.written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,reader,result)) ==
          (pack(source),Buffer.pack(~F32,written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next))) : Reader & Storage.Buffer<F32>}
      %Buffer.remodeled(~F32,Storage.write(~F32,position,value,Buffer.pack(~F32,result))) :
        {Kernel.written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,pack(source),_) ==
          (pack(source),Buffer.pack(~F32,written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next))) : Reader & Storage.Buffer<F32>}
      written(~Reader,~Source,~pack,~read,~sample,~matches,rest,1n+index,(position + 1 : U32),work,ranges,source,next)

def written_content(~Source: Data,~sample: Source -> Kernel.Context -> Nat -> F32,remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,
  +ranges: List<&2,Grouped.Range>,+source: Source,+result: Buffer.Model<F32>) ->
  {Buffer.content(~F32,written_model(~Source,~sample,remaining,index,position,work,ranges,source,result)) ==
    Buffer.Content{Tree.write_sequence(~F32,values(~Source,~sample,remaining,index,work,ranges,source),position,Buffer.content_tree(~F32,Buffer.content(~F32,result))),
      Buffer.content_length(~F32,Buffer.content(~F32,result))} : Buffer.Content<F32>}:
  match remaining:
    case 0n:
      Buffer.Model{tree,length,certificate} = result
      {==}
    case 1n+ +rest:
      +value = element(~Source,~sample,work,index,ranges,source)
      +next = Buffer.remodel(~F32,Storage.write(~F32,position,value,Buffer.pack(~F32,result)))
      +stepped = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,next),Buffer.contents(~F32,Storage.write(~F32,position,value,Buffer.pack(~F32,result))),
        Buffer.Content{Tree.write(F32,Buffer.content_tree(~F32,Buffer.content(~F32,result)),position,value),Buffer.content_length(~F32,Buffer.content(~F32,result))},
        Buffer.remodel_content(~F32,Storage.write(~F32,position,value,Buffer.pack(~F32,result))),Buffer.write_value_packed(~F32,position,value,result))
      %stepped :
        {Buffer.content(~F32,written_model(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next)) ==
          Buffer.Content{Tree.write_sequence(~F32,values(~Source,~sample,rest,1n+index,work,ranges,source),(position + 1 : U32),Buffer.content_tree(~F32,_)),
            Buffer.content_length(~F32,_)} : Buffer.Content<F32>}
      written_content(~Source,~sample,rest,1n+index,(position + 1 : U32),work,ranges,source,next)

def bias_written(+work: Kernel.Work,+source: Tree.Tree<F32>,+result: Buffer.Model<F32>) -> Buffer.Model<F32>:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  written_model(~Tree.Tree<F32>,~bias_sample,o,0n,0,work,Grouped.ranges(rows,columns,g),source,result)

def bias(+work: Kernel.Work,+source: Tree.Tree<F32>,+result: Buffer.Model<F32>) ->
  {Kernel.bias(work,Tree.pack(F32,source),Buffer.pack(~F32,result)) == (Tree.pack(F32,source),Buffer.pack(~F32,bias_written(work,source,result))) : Array<F32> & Storage.Buffer<F32>}:
  Kernel.Work{n,c,+o,h,w,k,s,p,+rows,+columns,+g} = work
  written(~Array<F32>,~Tree.Tree<F32>,~bias_pack,~Kernel.bias_at,~bias_sample,~bias_cell,o,0n,0,work,Grouped.ranges(rows,columns,g),source,result)

def bias_content(+work: Kernel.Work,+source: Tree.Tree<F32>,+result: Buffer.Model<F32>) ->
  {Buffer.content(~F32,bias_written(work,source,result)) ==
    Buffer.Content{Tree.write_sequence(~F32,bias_values(work,source),0,Buffer.content_tree(~F32,Buffer.content(~F32,result))),Buffer.content_length(~F32,Buffer.content(~F32,result))} :
    Buffer.Content<F32>}:
  Kernel.Work{n,c,+o,h,w,k,s,p,+rows,+columns,+g} = work
  written_content(~Tree.Tree<F32>,~bias_sample,o,0n,0,work,Grouped.ranges(rows,columns,g),source,result)
