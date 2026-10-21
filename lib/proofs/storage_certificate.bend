# A flat live certificate: depth plus one equality word. Its equation identifies
# the owner with a balanced normalization of its proof-only tree observation.
# Proof trees occur only under equality rewrites and are erased from native code.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_witness.bend as Witness

def first(-Element: Data,tree: Tree.Tree<Element>) -> Element:
  match tree:
    case Tree.Leaf{value}: value
    case Tree.Node{left,right}: first(Element,left)

def left(-Element: Data,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match tree:
    case Tree.Leaf{value}: Tree.Leaf{value}
    case Tree.Node{left,right}: left

def right(-Element: Data,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match tree:
    case Tree.Leaf{value}: Tree.Leaf{value}
    case Tree.Node{left,right}: right

def normalized(-Element: Data,depth: Nat,+tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match depth:
    case 0n: Tree.Leaf{first(Element,tree)}
    case 1n+ +rest:
      Tree.Node{normalized(Element,rest,left(Element,tree)),normalized(Element,rest,right(Element,tree))}

def normalized_witness(-Element: Data,+depth: Nat,-tree: Tree.Tree<Element>) ->
  Witness.Witness(Element,depth,Tree.pack(Element,normalized(Element,depth,tree))):
  match depth:
    case 0n: Witness.Leaf{first(Element,tree),{==}}
    case 1n+ +rest:
      Witness.Node{Tree.pack(Element,normalized(Element,rest,left(Element,tree))),
        Tree.pack(Element,normalized(Element,rest,right(Element,tree))),{==},
        normalized_witness(Element,rest,left(Element,tree)),normalized_witness(Element,rest,right(Element,tree))}

def restored(-Element: Data,+depth: Nat,-storage: Array<Element>,witness: Witness.Witness(Element,depth,storage)) ->
  {Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) == storage : Array<Element>}:
  match depth witness:
    case 0n Witness.Leaf{value,owner}:
      %Equal.sym(Array<Element>,storage,ALeaf{value},owner) :
        {Tree.pack(Element,normalized(Element,0n,Tree.reflect(Element,_))) == _ : Array<Element>}
      {==}
    case 1n+ +rest Witness.Node{left,right,owner,left_balanced,right_balanced}:
      %Equal.sym(Array<Element>,storage,ANode{left,right},owner) :
        {Tree.pack(Element,normalized(Element,1n+rest,Tree.reflect(Element,_))) == _ : Array<Element>}
      %restored(Element,rest,left,left_balanced) :
        {ANode{Tree.pack(Element,normalized(Element,rest,Tree.reflect(Element,left))),
          Tree.pack(Element,normalized(Element,rest,Tree.reflect(Element,right)))} == ANode{_,right} : Array<Element>}
      %restored(Element,rest,right,right_balanced) :
        {ANode{Tree.pack(Element,normalized(Element,rest,Tree.reflect(Element,left))),
          Tree.pack(Element,normalized(Element,rest,Tree.reflect(Element,right)))} ==
          ANode{Tree.pack(Element,normalized(Element,rest,Tree.reflect(Element,left))),_} : Array<Element>}
      {==}

type Certificate<-Element: Data,-storage: Array<Element>> is Data:
  Certificate{depth: Nat,equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>}}

def witness_at(-Element: Data,+depth: Nat,-storage: Array<Element>,
  equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>}) ->
  Witness.Witness(Element,depth,storage):
  %Equal.sym(Array<Element>,storage,Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))),equation) :
    Witness.Witness(Element,depth,_)
  normalized_witness(Element,depth,Tree.reflect(Element,storage))

def witness(-Element: Data,-storage: Array<Element>,certificate: Certificate<Element,storage>) -> Witness.ArrayWitness<Element,storage>:
  Certificate{+depth,equation} = certificate
  %Equal.sym(Array<Element>,storage,Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))),equation) :
    Witness.ArrayWitness<Element,_>
  Witness.ArrayWitness{depth,normalized_witness(Element,depth,Tree.reflect(Element,storage))}

def fresh_equation(-Element: Data,+depth: Nat,value: Element) ->
  {Array.new(Element,depth,value) == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Array.new(Element,depth,value)))) : Array<Element>}:
  %restored(Element,depth,Array.new(Element,depth,value),Witness.fresh(Element,depth,value)) :
    {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Array.new(Element,depth,value)))) : Array<Element>}
  {==}

def allocated(-Element: Data,+depth: Nat,value: Element) -> Certificate<Element,Array.new(Element,depth,value)>:
  Certificate{depth,fresh_equation(Element,depth,value)}

def family(~Element: Data,-storage: Array<Element>) -> Data:
  Certificate<Element,storage>

def written(~Element: Data,+depth: Nat,values: List<&2,Element>,+offset: U32,-storage: Array<Element>,
  evidence: Witness.Witness(Element,depth,storage)) -> Witness.Witness(Element,depth,Traversal.write_list(~Element,values,offset,storage)):
  match values:
    case []: evidence
    case head <> tail:
      written(~Element,depth,tail,(offset + 1 : U32),Array.set(Element,storage,offset,head),Witness.write(Element,depth,storage,offset,head,evidence))

def written_equation(~Element: Data,+depth: Nat,values: List<&2,Element>,offset: U32,-storage: Array<Element>,
  equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>}) ->
  {Traversal.write_list(~Element,values,offset,storage) ==
    Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Traversal.write_list(~Element,values,offset,storage)))) : Array<Element>}:
  %restored(Element,depth,Traversal.write_list(~Element,values,offset,storage),
    written(~Element,depth,values,offset,storage,witness_at(Element,depth,storage,equation))) :
    {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Traversal.write_list(~Element,values,offset,storage)))) : Array<Element>}
  {==}

def write_list(~Element: Data,values: List<&2,Element>,offset: U32,-storage: Array<Element>,certificate: Certificate<Element,storage>) ->
  Certificate<Element,Traversal.write_list(~Element,values,offset,storage)>:
  Certificate{+depth,equation} = certificate
  Certificate{depth,written_equation(~Element,depth,values,offset,storage,equation)}

def set_equation(~Element: Data,+depth: Nat,+index: U32,-value: Element,-storage: Array<Element>,
  equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>}) ->
  {Array.set(Element,storage,index,value) ==
    Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Array.set(Element,storage,index,value)))) : Array<Element>}:
  %restored(Element,depth,Array.set(Element,storage,index,value),
    Witness.write(Element,depth,storage,index,value,witness_at(Element,depth,storage,equation))) :
    {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Array.set(Element,storage,index,value)))) : Array<Element>}
  {==}

# One value written; the balance depth is unchanged whatever the value.
def set(~Element: Data,+index: U32,-value: Element,-storage: Array<Element>,certificate: Certificate<Element,storage>) ->
  Certificate<Element,Array.set(Element,storage,index,value)>:
  Certificate{+depth,equation} = certificate
  Certificate{depth,set_equation(~Element,depth,index,value,storage,equation)}

# Transport a storage certificate across the read-only native array operation.
def retained(~Element: Data,+depth: Nat,-storage: Array<Element>,-value: Element,-result: Array<Element> & Element,
  +same: {(storage,value) == result : Array<Element> & Element},
  +equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>}) ->
  Certificate<Element,Pair.fst(Array<Element>,Element,result)>:
  Certificate{depth,
    %Equal.cong(Array<Element> & Element,Array<Element>,pair => Pair.fst(Array<Element>,Element,pair),(storage,value),result,same) :
      {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,_))) : Array<Element>}
    equation}

def read_owner(~Element: Data,-storage: Array<Element>,+index: U32,
  read: Witness.Read<Element,storage,Array.get(Element,storage,index)>) ->
  {storage == Pair.fst(Array<Element>,Element,Array.get(Element,storage,index)) : Array<Element>}:
  Witness.Read{value,+same} = read
  Equal.cong(Array<Element> & Element,Array<Element>,pair => Pair.fst(Array<Element>,Element,pair),(storage,value),Array.get(Element,storage,index),
    Equal.sym(Array<Element> & Element,Array.get(Element,storage,index),(storage,value),same))

def read_equation(~Element: Data,+depth: Nat,-storage: Array<Element>,+index: U32,
  +equation: {storage == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>},
  read: Witness.Read<Element,storage,Array.get(Element,storage,index)>) ->
  {Pair.fst(Array<Element>,Element,Array.get(Element,storage,index)) == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,Pair.fst(Array<Element>,Element,Array.get(Element,storage,index))))) : Array<Element>}:
  %read_owner(~Element,storage,index,read) :
    {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,_))) : Array<Element>}
  equation

def read_kept(~Element: Data,-storage: Array<Element>,+index: U32,certificate: Certificate<Element,storage>) ->
  Certificate<Element,Pair.fst(Array<Element>,Element,Array.get(Element,storage,index))>:
  Certificate{+depth,+equation} = certificate
  Certificate{depth,
    %read_owner(~Element,storage,index,Witness.observed(Element,storage,index,Witness.ArrayWitness{depth,witness_at(Element,depth,storage,equation)})) :
      {_ == Tree.pack(Element,normalized(Element,depth,Tree.reflect(Element,_))) : Array<Element>}
    equation}
