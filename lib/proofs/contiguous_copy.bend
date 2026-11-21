# Exact stores of the shared contiguous copy, including preservation of its
# input owner. No bounds premise is needed for this execution refinement.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes

def writes(~Element: Data,count: Nat,+input: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{(destination + position : U32),Tree.read(Element,input,(source + position : U32))} <>
        writes(~Element,rest,input,(position + 1 : U32),source,destination)

def traversal(~Element: Data,+count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32) ->
  {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
      Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,count,
        Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),position,Traversal.Offsets{source,destination}})) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,count,input,position,source,destination),output))) : Array<Element> & Array<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),(source + position : U32)),
        (Tree.pack(Element,input),Tree.read(Element,input,(source + position : U32))),StorageLaws.read(Element,input,(source + position : U32))) :
        {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
          Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,rest,
            Traversal.store(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.destination_offset,position,Traversal.Offsets{source,destination},Tree.pack(Element,output),_))) ==
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,1n+rest,input,position,source,destination),output))) : Array<Element> & Array<Element>}
      %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),(destination + position : U32),Tree.read(Element,input,(source + position : U32))),
        Tree.pack(Element,Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32)))),
        StorageLaws.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32)))) :
        {Traversal.finish(~Array<Element>,~Element,~Traversal.Offsets,
          Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,rest,
            Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),Traversal.Offsets{source,destination}})) ==
          (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,1n+rest,input,position,source,destination),output))) : Array<Element> & Array<Element>}
      traversal(~Element,rest,input,Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32))),
        (position + 1 : U32),source,destination)

def copy(~Element: Data,+count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+source: U32,+destination: U32) ->
  {Traversal.copy(~Element,count,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,writes(~Element,count,input,0,source,destination),output))) : Array<Element> & Array<Element>}:
  traversal(~Element,count,input,output,0,source,destination)

