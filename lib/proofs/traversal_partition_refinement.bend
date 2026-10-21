# Trees of parts: maps, folds and facts about every part, gathers over them,
# and the model split that names both halves.
import Base
import ../traversal_partition.bend as Partition
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes

# Every part replaced by what one rule gives it; the splits stay.
def map(~from: Quant,~to: Quant,~Before: Kind(from),~After: Kind(to),~Details: Data,~Result: Data,~Context: Data,
  ~part: @+context: Context -> @+offset: U32 -> @+span: U32 -> @+details: Details -> @block: Before -> After,
  ~detail: @+context: Context -> @+details: Details -> Result,
  +context: Context,tree: Partition.Tree<from,Before,Details>) -> Partition.Tree<to,After,Result>:
  match tree:
    case Partition.Leaf{block,+offset,+span,+details}: Partition.Leaf{part(context,offset,span,details,block),offset,span,detail(context,details)}
    case Partition.Fork{left,right}: Partition.Fork{map(~from,~to,~Before,~After,~Details,~Result,~Context,~part,~detail,context,left),map(~from,~to,~Before,~After,~Details,~Result,~Context,~part,~detail,context,right)}

def keep(~Details: Data,+context: Unit,+details: Details) -> Details:
  details

def through(~from: Quant,~to: Quant,~Before: Kind(from),~After: Kind(to),~Details: Data,~change: Before -> After,
  +context: Unit,+offset: U32,+span: U32,+details: Details,block: Before) -> After:
  change(block)

# The same tree with every block in another representation.
def carry(~from: Quant,~to: Quant,~Before: Kind(from),~After: Kind(to),~Details: Data,~change: Before -> After,tree: Partition.Tree<from,Before,Details>) ->
  Partition.Tree<to,After,Details>:
  map(~from,~to,~Before,~After,~Details,~Details,~Unit,~through(~from,~to,~Before,~After,~Details,~change),~keep(~Details),Unit{},tree)

# One value from every part, joined at the splits.
def reduce(~Block: Data,~Details: Data,~Result: Data,~Context: Data,
  ~part: @+context: Context -> @+offset: U32 -> @+span: U32 -> @+details: Details -> @block: Block -> Result,
  ~join: Result -> Result -> Result,+context: Context,tree: Partition.Tree<&2,Block,Details>) -> Result:
  match tree:
    case Partition.Leaf{block,+offset,+span,+details}: part(context,offset,span,details,block)
    case Partition.Fork{left,right}: join(reduce(~Block,~Details,~Result,~Context,~part,~join,context,left),reduce(~Block,~Details,~Result,~Context,~part,~join,context,right))

# A fact about every part.
def All(~ownership: Quant,~Block: Kind(ownership),~Details: Data,~Context: Data,
  ~Part: @+context: Context -> @+offset: U32 -> @+span: U32 -> @+details: Details -> @block: Block -> Data,
  +context: Context,tree: Partition.Tree<ownership,Block,Details>) -> Data:
  match tree:
    case Partition.Leaf{block,+offset,+span,+details}: Part(context,offset,span,details,block)
    case Partition.Fork{left,right}: Shape.Both<All(~ownership,~Block,~Details,~Context,~Part,context,left),All(~ownership,~Block,~Details,~Context,~Part,context,right)>

def pack_values(~Output: Data,data: Tree.Tree<Output>) -> Array<Output>:
  Tree.pack(Output,data)

def represent(~Output: Data,tree: Partition.Tree<&2,Tree.Tree<Output>,U32>) -> Partition.Tree<&1,Array<Output>,U32>:
  carry(~&2,~&1,~Tree.Tree<Output>,~Array<Output>,~U32,~pack_values(~Output),tree)

# A gather over packed blocks is the packed gather of the models, once the
# copy of one block is.
def gathered(~ownership: Quant,~Native: Kind(ownership),~Model: Data,~pack: Model -> Native,
  ~native_copy: U32 -> U32 -> U32 -> U32 -> Native -> Native -> Native,~model_copy: U32 -> U32 -> U32 -> U32 -> Model -> Model -> Model,
  ~leaf: @+rows: U32 -> @+offset: U32 -> @+positions: U32 -> @+span: U32 -> @+input: Model -> @+output: Model ->
    {native_copy(rows,offset,positions,span,pack(input),pack(output)) == pack(model_copy(rows,offset,positions,span,input,output)) : Native},
  +tree: Partition.Tree<&2,Model,U32>,+output: Model,+positions: U32) ->
  {Partition.gather(~ownership,~Native,~native_copy,carry(~&2,~ownership,~Model,~Native,~U32,~pack,tree),pack(output),positions) == pack(Partition.gather(~&2,~Model,~model_copy,tree,output,positions)) : Native}:
  match tree:
    case Partition.Leaf{+input,+offset,+span,+rows}: leaf(rows,offset,positions,span,input,output)
    case Partition.Fork{+left,+right}:
      %Equal.sym(Native,Partition.gather(~ownership,~Native,~native_copy,carry(~&2,~ownership,~Model,~Native,~U32,~pack,left),pack(output),positions),pack(Partition.gather(~&2,~Model,~model_copy,left,output,positions)),gathered(~ownership,~Native,~Model,~pack,~native_copy,~model_copy,~leaf,left,output,positions)) :
        {Partition.gather(~ownership,~Native,~native_copy,carry(~&2,~ownership,~Model,~Native,~U32,~pack,right),_,positions) == pack(Partition.gather(~&2,~Model,~model_copy,right,Partition.gather(~&2,~Model,~model_copy,left,output,positions),positions)) : Native}
      gathered(~ownership,~Native,~Model,~pack,~native_copy,~model_copy,~leaf,right,Partition.gather(~&2,~Model,~model_copy,left,output,positions),positions)

# A gather whose block copy applies a list of writes applies the writes of
# all blocks, in order.
def write_copy(~T: Data,~writes: @channels: U32 -> @+offset: U32 -> @+positions: U32 -> @+span: U32 -> @input: Tree.Tree<T> -> List<&2,Writes.Assignment<T>>,rows: U32,offset: U32,positions: U32,span: U32,input: Tree.Tree<T>,output: Tree.Tree<T>) -> Tree.Tree<T>:
  +offset = offset
  +positions = positions
  +span = span
  Writes.pure(T,writes(rows,offset,positions,span,input),output)

def writes_part(~T: Data,~writes: @channels: U32 -> @+offset: U32 -> @+positions: U32 -> @+span: U32 -> @input: Tree.Tree<T> -> List<&2,Writes.Assignment<T>>,+positions: U32,+offset: U32,+span: U32,+rows: U32,block: Tree.Tree<T>) -> List<&2,Writes.Assignment<T>>:
  writes(rows,offset,positions,span,block)

def appended(~T: Data,first: List<&2,Writes.Assignment<T>>,second: List<&2,Writes.Assignment<T>>) -> List<&2,Writes.Assignment<T>>:
  List.append(&2,Writes.Assignment<T>,first,second)

def tree_writes(~T: Data,~writes: @channels: U32 -> @+offset: U32 -> @+positions: U32 -> @+span: U32 -> @input: Tree.Tree<T> -> List<&2,Writes.Assignment<T>>,tree: Partition.Tree<&2,Tree.Tree<T>,U32>,+positions: U32) -> List<&2,Writes.Assignment<T>>:
  reduce(~Tree.Tree<T>,~U32,~List<&2,Writes.Assignment<T>>,~U32,~writes_part(~T,~writes),~appended(~T),positions,tree)

def gather_written(~T: Data,~writes: @channels: U32 -> @+offset: U32 -> @+positions: U32 -> @+span: U32 -> @input: Tree.Tree<T> -> List<&2,Writes.Assignment<T>>,+tree: Partition.Tree<&2,Tree.Tree<T>,U32>,+output: Tree.Tree<T>,+positions: U32) ->
  {Partition.gather(~&2,~Tree.Tree<T>,~write_copy(~T,~writes),tree,output,positions) == Writes.pure(T,tree_writes(~T,~writes,tree,positions),output) : Tree.Tree<T>}:
  match tree:
    case Partition.Leaf{input,offset,span,rows}: {==}
    case Partition.Fork{+left,+right}:
      %Equal.sym(Tree.Tree<T>,Partition.gather(~&2,~Tree.Tree<T>,~write_copy(~T,~writes),left,output,positions),Writes.pure(T,tree_writes(~T,~writes,left,positions),output),gather_written(~T,~writes,left,output,positions)) :
        {Partition.gather(~&2,~Tree.Tree<T>,~write_copy(~T,~writes),right,_,positions) == Writes.pure(T,List.append(&2,Writes.Assignment<T>,tree_writes(~T,~writes,left,positions),tree_writes(~T,~writes,right,positions)),output) : Tree.Tree<T>}
      %Equal.sym(Tree.Tree<T>,Writes.pure(T,List.append(&2,Writes.Assignment<T>,tree_writes(~T,~writes,left,positions),tree_writes(~T,~writes,right,positions)),output),
        Writes.pure(T,tree_writes(~T,~writes,right,positions),Writes.pure(T,tree_writes(~T,~writes,left,positions),output)),
        Writes.concatenated_writes(T,tree_writes(~T,~writes,left,positions),tree_writes(~T,~writes,right,positions),output)) :
        {Partition.gather(~&2,~Tree.Tree<T>,~write_copy(~T,~writes),right,Writes.pure(T,tree_writes(~T,~writes,left,positions),output),positions) == _ : Tree.Tree<T>}
      gather_written(~T,~writes,right,Writes.pure(T,tree_writes(~T,~writes,left,positions),output),positions)

# A model split names both children; the native split returns their owners.
type Split<-Input: Data> is Data:
  Split{left: Input,right: Input}

def first(~Input: Data,pair: Split<Input>) -> Input:
  Split{left,right} = pair
  left

def second(~Input: Data,pair: Split<Input>) -> Input:
  Split{left,right} = pair
  right

def split_eta(~Input: Data,+pair: Split<Input>) -> {pair == Split{first(~Input,pair),second(~Input,pair)} : Split<Input>}:
  Split{left,right} = pair
  {==}

def represent_pair(~Native: Type,~Input: Data,~represent_input: Input -> Native,pair: Split<Input>) -> Native & Native:
  Split{left,right} = pair
  (represent_input(left),represent_input(right))

