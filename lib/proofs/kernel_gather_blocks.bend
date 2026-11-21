# The actual channel/block schedule. The inner traversal reuses the library's
# exact contiguous-copy theorem; concatenation preserves the order of stores.
import Base
import ../kernel_gather.bend as Gather
import ./storage_tree_model.bend as Tree
import ./contiguous_copy.bend as Contiguous
import ./write_program.bend as Writes

def writes(~Element: Data,count: Nat,+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<Element>,Contiguous.writes(~Element,U32.to_nat(width),input,0,(channel * span + source_start : U32),(channel * spatial + offset : U32)),
        writes(~Element,rest,(channel + 1 : U32),width,offset,spatial,span,source_start,input))

def channel(~Element: Data,+channel: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Gather.gather_channel(~Element,channel,width,offset,spatial,span,source_start,(Tree.pack(Element,input),Tree.pack(Element,output))) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(channel * span + source_start : U32),(channel * spatial + offset : U32)),output))) : Array<Element> & Array<Element>}:
  Contiguous.copy(~Element,U32.to_nat(width),input,output,(channel * span + source_start : U32),(channel * spatial + offset : U32))

def channels(~Element: Data,+count: Nat,+row: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Gather.gather_channels(~Element,count,row,width,offset,spatial,span,source_start,(Tree.pack(Element,input),Tree.pack(Element,output))) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,count,row,width,offset,spatial,span,source_start,input),output))) : Array<Element> & Array<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Array<Element> & Array<Element>,Gather.gather_channel(~Element,row,width,offset,spatial,span,source_start,(Tree.pack(Element,input),Tree.pack(Element,output))),
        (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),output))),
        channel(~Element,row,width,offset,spatial,span,source_start,input,output)) :
        {Gather.gather_channels(~Element,rest,(row + 1 : U32),width,offset,spatial,span,source_start,_) ==
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,1n+rest,row,width,offset,spatial,span,source_start,input),output))) : Array<Element> & Array<Element>}
      %Equal.sym(Tree.Tree<Element>,Writes.pure(Element,writes(~Element,1n+rest,row,width,offset,spatial,span,source_start,input),output),
        Writes.pure(Element,writes(~Element,rest,(row + 1 : U32),width,offset,spatial,span,source_start,input),
          Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),output)),
        Writes.concatenated_writes(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),
          writes(~Element,rest,(row + 1 : U32),width,offset,spatial,span,source_start,input),output)) :
        {Gather.gather_channels(~Element,rest,(row + 1 : U32),width,offset,spatial,span,source_start,
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),output)))) ==
          (Tree.pack(Element,input),Tree.pack(Element,_)) : Array<Element> & Array<Element>}
      channels(~Element,rest,(row + 1 : U32),width,offset,spatial,span,source_start,input,
        Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),output))

def block_writes(~Element: Data,channels: U32,+offset: U32,+spatial: U32,+span: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  writes(~Element,U32.to_nat(channels),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,0,input)

def block(~Element: Data,+count: U32,+offset: U32,+spatial: U32,+span: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Gather.gather_block(~Element,count,offset,spatial,span,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Tree.pack(Element,Writes.pure(Element,block_writes(~Element,count,offset,spatial,span,input),output)) : Array<Element>}:
  %Equal.sym(Array<Element> & Array<Element>,Gather.gather_channels(~Element,U32.to_nat(count),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,0,(Tree.pack(Element,input),Tree.pack(Element,output))),
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,block_writes(~Element,count,offset,spatial,span,input),output))),
    channels(~Element,U32.to_nat(count),0,Gather.gather_width(offset,spatial,span),offset,spatial,span,0,input,output)) :
    {Gather.gathered_output(~Element,_) == Tree.pack(Element,Writes.pure(Element,block_writes(~Element,count,offset,spatial,span,input),output)) : Array<Element>}
  {==}
