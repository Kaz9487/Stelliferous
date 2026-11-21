# Regrouped blocks, by their models. A tree of represented blocks regrouped
# into the shape of ranges is the representation of its block models
# regrouped the same way: the blocks leave the tree in order, every range
# takes the next, and a range past the last takes the vacant block.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./traversal_partition_refinement.bend as PartitionLaws
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# The models of the blocks of a tree, in order, before rest.
def queued(~Model: Data,tree: Partition.Tree<&2,Model,Unit>,rest: List<&2,Model>) -> List<&2,Model>:
  match tree:
    case Partition.Leaf{model,first,count,details}: model <> rest
    case Partition.Fork{left,right}: queued(~Model,left,queued(~Model,right,rest))

def head(~Model: Data,~Fill: Data,~vacant: Fill -> Model,+fill: Fill,queue: List<&2,Model>) -> Model:
  match queue:
    case Con{model,rest}: model
    case Nil{}: vacant(fill)

def tail(~Model: Data,queue: List<&2,Model>) -> List<&2,Model>:
  match queue:
    case Con{model,rest}: rest
    case Nil{}: []

# The queue after the ranges take their blocks.
def taken(~Model: Data,layout: Pieces.Ranges,queue: List<&2,Model>) -> List<&2,Model>:
  match layout:
    case Pieces.Range{first,count}: tail(~Model,queue)
    case Pieces.Split{left,right}: taken(~Model,right,taken(~Model,left,queue))

# The ranges, each with the block it takes.
def built(~Model: Data,~Fill: Data,~vacant: Fill -> Model,layout: Pieces.Ranges,+fill: Fill,+queue: List<&2,Model>) -> Partition.Tree<&2,Model,Unit>:
  match layout:
    case Pieces.Range{first,count}: Partition.Leaf{head(~Model,~Fill,~vacant,fill,queue),first,count,Unit{}}
    case Pieces.Split{+left,right}:
      Partition.Fork{built(~Model,~Fill,~vacant,left,fill,queue),built(~Model,~Fill,~vacant,right,fill,taken(~Model,left,queue))}

# The block models of a tree in the shape of ranges.
def regrouped(~Model: Data,~Fill: Data,~vacant: Fill -> Model,+layout: Pieces.Ranges,+fill: Fill,tree: Partition.Tree<&2,Model,Unit>) -> Partition.Tree<&2,Model,Unit>:
  built(~Model,~Fill,~vacant,layout,fill,queued(~Model,tree,[]))

# ---- The blocks are their models ----

def tree_of(~Block: Type,~Model: Data,~represent: Model -> Block,tree: Partition.Tree<&2,Model,Unit>) -> Partition.Tree<&1,Block,Unit>:
  PartitionLaws.carry(~&2,~&1,~Model,~Block,~Unit,~represent,tree)

def queue_of(~Block: Type,~Model: Data,~represent: Model -> Block,models: List<&2,Model>) -> Pieces.Queue<Block>:
  match models:
    case Nil{}: Pieces.Empty{}
    case Con{model,rest}: Pieces.Queued{represent(model),queue_of(~Block,~Model,~represent,rest)}

def queued_same(~Block: Type,~Model: Data,~represent: Model -> Block,+tree: Partition.Tree<&2,Model,Unit>,+rest: List<&2,Model>) ->
  {Pieces.queued(~Block,tree_of(~Block,~Model,~represent,tree),queue_of(~Block,~Model,~represent,rest)) == queue_of(~Block,~Model,~represent,queued(~Model,tree,rest)) : Pieces.Queue<Block>}:
  match tree:
    case Partition.Leaf{model,first,count,Unit{}}: {==}
    case Partition.Fork{+left,+right}:
      %Equal.sym(Pieces.Queue<Block>,Pieces.queued(~Block,tree_of(~Block,~Model,~represent,right),queue_of(~Block,~Model,~represent,rest)),
        queue_of(~Block,~Model,~represent,queued(~Model,right,rest)),queued_same(~Block,~Model,~represent,right,rest)) :
        {Pieces.queued(~Block,tree_of(~Block,~Model,~represent,left),_) == queue_of(~Block,~Model,~represent,queued(~Model,left,queued(~Model,right,rest))) : Pieces.Queue<Block>}
      queued_same(~Block,~Model,~represent,left,queued(~Model,right,rest))

def dequeued_same(~Block: Type,~Model: Data,~Fill: Data,~vacant: Fill -> Block,~represent: Model -> Block,~vacant_model: Fill -> Model,
  ~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},+fill: Fill,+first: U32,+count: U32,+queue: List<&2,Model>) ->
  {Pieces.dequeued(~Block,~Fill,~vacant,fill,first,count,queue_of(~Block,~Model,~represent,queue)) ==
    (queue_of(~Block,~Model,~represent,tail(~Model,queue)),Partition.Leaf{represent(head(~Model,~Fill,~vacant_model,fill,queue)),first,count,Unit{}}) :
    Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>}:
  match queue:
    case Con{model,rest}: {==}
    case Nil{}:
      %vacant_same(fill) :
        {(Pieces.Empty{},Partition.Leaf{vacant(fill),first,count,Unit{}}) == (Pieces.Empty{},Partition.Leaf{_,first,count,Unit{}}) : Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>}
      {==}

def build_same(~Block: Type,~Model: Data,~Fill: Data,~vacant: Fill -> Block,~represent: Model -> Block,~vacant_model: Fill -> Model,
  ~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},+layout: Pieces.Ranges,+fill: Fill,+queue: List<&2,Model>) ->
  {Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),layout,fill,queue_of(~Block,~Model,~represent,queue)) ==
    (queue_of(~Block,~Model,~represent,taken(~Model,layout,queue)),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,layout,fill,queue))) :
    Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>}:
  match layout:
    case Pieces.Range{+first,+count}: dequeued_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,fill,first,count,queue)
    case Pieces.Split{+left,+right}:
      +rest = taken(~Model,left,queue)
      %Equal.sym(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>,
        Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),left,fill,queue_of(~Block,~Model,~represent,queue)),
        (queue_of(~Block,~Model,~represent,rest),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,left,fill,queue))),
        build_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,left,fill,queue)) :
        {Storage.bind_pair(~Pieces.Queue<Block>,~Partition.Tree<&1,Block,Unit>,~(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>),_,state => left =>
            Storage.bind_pair(~Pieces.Queue<Block>,~Partition.Tree<&1,Block,Unit>,~(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>),
              Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),right,fill,state),state => right => (state,Partition.Fork{left,right}))) ==
          (queue_of(~Block,~Model,~represent,taken(~Model,right,rest)),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,Pieces.Split{left,right},fill,queue))) :
          Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>}
      %Equal.sym(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>,
        Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),right,fill,queue_of(~Block,~Model,~represent,rest)),
        (queue_of(~Block,~Model,~represent,taken(~Model,right,rest)),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,right,fill,rest))),
        build_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,right,fill,rest)) :
        {Storage.bind_pair(~Pieces.Queue<Block>,~Partition.Tree<&1,Block,Unit>,~(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>),_,state => right =>
            (state,Partition.Fork{tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,left,fill,queue)),right})) ==
          (queue_of(~Block,~Model,~represent,taken(~Model,right,rest)),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,Pieces.Split{left,right},fill,queue))) :
          Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>}
      {==}

# Represented blocks regrouped are their models regrouped, given the vacant
# block is the representation of a model.
def regrouped_same(~Block: Type,~Model: Data,~Fill: Data,~vacant: Fill -> Block,~represent: Model -> Block,~vacant_model: Fill -> Model,
  ~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},+layout: Pieces.Ranges,+fill: Fill,+tree: Partition.Tree<&2,Model,Unit>) ->
  {Pieces.blocks_regrouped(~Block,~Fill,~vacant,layout,fill,tree_of(~Block,~Model,~represent,tree)) ==
    tree_of(~Block,~Model,~represent,regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)) : Partition.Tree<&1,Block,Unit>}:
  +queue = queued(~Model,tree,[])
  %Equal.sym(Pieces.Queue<Block>,Pieces.queued(~Block,tree_of(~Block,~Model,~represent,tree),queue_of(~Block,~Model,~represent,[])),queue_of(~Block,~Model,~represent,queue),
    queued_same(~Block,~Model,~represent,tree,[])) :
    {Pair.snd(Pieces.Queue<Block>,Partition.Tree<&1,Block,Unit>,Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),layout,fill,_)) ==
      tree_of(~Block,~Model,~represent,regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)) : Partition.Tree<&1,Block,Unit>}
  %Equal.sym(Pieces.Queue<Block> & Partition.Tree<&1,Block,Unit>,
    Pieces.build(~Block,~Pieces.Queue<Block>,~Fill,~Pieces.dequeued(~Block,~Fill,~vacant),layout,fill,queue_of(~Block,~Model,~represent,queue)),
    (queue_of(~Block,~Model,~represent,taken(~Model,layout,queue)),tree_of(~Block,~Model,~represent,built(~Model,~Fill,~vacant_model,layout,fill,queue))),
    build_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,layout,fill,queue)) :
    {Pair.snd(Pieces.Queue<Block>,Partition.Tree<&1,Block,Unit>,_) ==
      tree_of(~Block,~Model,~represent,regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)) : Partition.Tree<&1,Block,Unit>}
  {==}

# ---- A regrouping keeps the blocks ----

# The number of ranges of a layout.
def ranges(layout: Pieces.Ranges) -> Nat:
  match layout:
    case Pieces.Range{first,count}: 1n
    case Pieces.Split{left,right}: Nat.add(ranges(left),ranges(right))

# The ranges of the leaves of a tree.
def tree_layout(~Model: Data,tree: Partition.Tree<&2,Model,Unit>) -> Pieces.Ranges:
  match tree:
    case Partition.Leaf{model,first,count,details}: Pieces.Range{first,count}
    case Partition.Fork{left,right}: Pieces.Split{tree_layout(~Model,left),tree_layout(~Model,right)}

def predecessor(value: Nat) -> Nat:
  match value:
    case 0n: 0n
    case 1n+rest: rest

# A queue with a block for a range is not empty.
def queue_held(+spare: Nat,+same: {1n+spare == 0n : Nat}) -> Empty:
  WordEncoding.false_true(Equal.cong(Nat,Bool,value => Nat.is_eq(value,0n),1n+spare,0n,same))

def length_empty(~Model: Data,+queue: List<&2,Model>,+empty: {List.length(&2,Model,queue) == 0n : Nat}) -> {queue == [] : List<&2,Model>}:
  match queue:
    case Nil{}: {==}
    case Con{+model,+rest}: Empty.absurd({Con{model,rest} == [] : List<&2,Model>},queue_held(List.length(&2,Model,rest),empty))

# The blocks of a tree in order number its ranges.
def queued_count(~Model: Data,+tree: Partition.Tree<&2,Model,Unit>,+rest: List<&2,Model>) ->
  {List.length(&2,Model,queued(~Model,tree,rest)) == Nat.add(ranges(tree_layout(~Model,tree)),List.length(&2,Model,rest)) : Nat}:
  match tree:
    case Partition.Leaf{model,first,count,details}: {==}
    case Partition.Fork{+left,+right}:
      +left_ranges = ranges(tree_layout(~Model,left))
      +right_ranges = ranges(tree_layout(~Model,right))
      Equal.trans(Nat,List.length(&2,Model,queued(~Model,left,queued(~Model,right,rest))),Nat.add(left_ranges,List.length(&2,Model,queued(~Model,right,rest))),Nat.add(Nat.add(left_ranges,right_ranges),List.length(&2,Model,rest)),
        queued_count(~Model,left,queued(~Model,right,rest)),
        Equal.trans(Nat,Nat.add(left_ranges,List.length(&2,Model,queued(~Model,right,rest))),Nat.add(left_ranges,Nat.add(right_ranges,List.length(&2,Model,rest))),Nat.add(Nat.add(left_ranges,right_ranges),List.length(&2,Model,rest)),
          Equal.cong(Nat,Nat,number => Nat.add(left_ranges,number),List.length(&2,Model,queued(~Model,right,rest)),Nat.add(right_ranges,List.length(&2,Model,rest)),queued_count(~Model,right,rest)),
          Equal.sym(Nat,Nat.add(Nat.add(left_ranges,right_ranges),List.length(&2,Model,rest)),Nat.add(left_ranges,Nat.add(right_ranges,List.length(&2,Model,rest))),
            Addition.associative(left_ranges,right_ranges,List.length(&2,Model,rest)))))

# The left ranges of a split, counted before the right ones and the spare.
def left_counted(+left: Pieces.Ranges,+right: Pieces.Ranges,+spare: Nat,+length: Nat,
  +counted: {Nat.add(Nat.add(ranges(left),ranges(right)),spare) == length : Nat}) -> {Nat.add(ranges(left),Nat.add(ranges(right),spare)) == length : Nat}:
  Equal.trans(Nat,Nat.add(ranges(left),Nat.add(ranges(right),spare)),Nat.add(Nat.add(ranges(left),ranges(right)),spare),length,
    Equal.sym(Nat,Nat.add(Nat.add(ranges(left),ranges(right)),spare),Nat.add(ranges(left),Nat.add(ranges(right),spare)),Addition.associative(ranges(left),ranges(right),spare)),counted)

# The ranges of a layout take a block each and leave the spare blocks.
def taken_count(~Model: Data,+layout: Pieces.Ranges,+queue: List<&2,Model>,+spare: Nat,
  +counted: {Nat.add(ranges(layout),spare) == List.length(&2,Model,queue) : Nat}) ->
  {List.length(&2,Model,taken(~Model,layout,queue)) == spare : Nat}:
  match layout:
    case Pieces.Range{+first,+count}:
      match queue:
        case Nil{}: Empty.absurd({List.length(&2,Model,taken(~Model,Pieces.Range{first,count},[])) == spare : Nat},queue_held(spare,counted))
        case Con{+model,+rest}: Equal.sym(Nat,spare,List.length(&2,Model,rest),Equal.cong(Nat,Nat,value => predecessor(value),1n+spare,1n+List.length(&2,Model,rest),counted))
    case Pieces.Split{+left,+right}:
      +more = Nat.add(ranges(right),spare)
      +rest_counted = taken_count(~Model,left,queue,more,left_counted(left,right,spare,List.length(&2,Model,queue),counted))
      taken_count(~Model,right,taken(~Model,left,queue),spare,Equal.sym(Nat,List.length(&2,Model,taken(~Model,left,queue)),more,rest_counted))

# The blocks the ranges of a layout take, then the blocks they leave, are the
# queue, when it holds a block for every range.
def built_taken(~Model: Data,~Fill: Data,~vacant: Fill -> Model,+layout: Pieces.Ranges,+fill: Fill,+queue: List<&2,Model>,+spare: Nat,
  +counted: {Nat.add(ranges(layout),spare) == List.length(&2,Model,queue) : Nat}) ->
  {queued(~Model,built(~Model,~Fill,~vacant,layout,fill,queue),taken(~Model,layout,queue)) == queue : List<&2,Model>}:
  match layout:
    case Pieces.Range{+first,+count}:
      match queue:
        case Nil{}:
          Empty.absurd({queued(~Model,built(~Model,~Fill,~vacant,Pieces.Range{first,count},fill,[]),taken(~Model,Pieces.Range{first,count},[])) == [] : List<&2,Model>},queue_held(spare,counted))
        case Con{+model,+rest}: {==}
    case Pieces.Split{+left,+right}:
      +more = Nat.add(ranges(right),spare)
      +more_counted = left_counted(left,right,spare,List.length(&2,Model,queue),counted)
      +rest = taken(~Model,left,queue)
      +right_counted = Equal.sym(Nat,List.length(&2,Model,rest),more,taken_count(~Model,left,queue,more,more_counted))
      %Equal.sym(List<&2,Model>,queued(~Model,built(~Model,~Fill,~vacant,right,fill,rest),taken(~Model,right,rest)),rest,built_taken(~Model,~Fill,~vacant,right,fill,rest,spare,right_counted)) :
        {queued(~Model,built(~Model,~Fill,~vacant,left,fill,queue),_) == queue : List<&2,Model>}
      built_taken(~Model,~Fill,~vacant,left,fill,queue,more,more_counted)

# A tree regrouped into its own ranges takes its blocks and leaves the rest.
def rebuilt_taken(~Model: Data,+tree: Partition.Tree<&2,Model,Unit>,+rest: List<&2,Model>) ->
  {taken(~Model,tree_layout(~Model,tree),queued(~Model,tree,rest)) == rest : List<&2,Model>}:
  match tree:
    case Partition.Leaf{model,first,count,details}: {==}
    case Partition.Fork{+left,+right}:
      Equal.trans(List<&2,Model>,taken(~Model,tree_layout(~Model,right),taken(~Model,tree_layout(~Model,left),queued(~Model,left,queued(~Model,right,rest)))),
        taken(~Model,tree_layout(~Model,right),queued(~Model,right,rest)),rest,
        Equal.cong(List<&2,Model>,List<&2,Model>,queue => taken(~Model,tree_layout(~Model,right),queue),taken(~Model,tree_layout(~Model,left),queued(~Model,left,queued(~Model,right,rest))),queued(~Model,right,rest),
          rebuilt_taken(~Model,left,queued(~Model,right,rest))),
        rebuilt_taken(~Model,right,rest))

# A tree regrouped into its own ranges is itself.
def rebuilt(~Model: Data,~Fill: Data,~vacant: Fill -> Model,+fill: Fill,+tree: Partition.Tree<&2,Model,Unit>,+rest: List<&2,Model>) ->
  {built(~Model,~Fill,~vacant,tree_layout(~Model,tree),fill,queued(~Model,tree,rest)) == tree : Partition.Tree<&2,Model,Unit>}:
  match tree:
    case Partition.Leaf{model,first,count,Unit{}}: {==}
    case Partition.Fork{+left,+right}:
      +queue = queued(~Model,left,queued(~Model,right,rest))
      %Equal.sym(Partition.Tree<&2,Model,Unit>,built(~Model,~Fill,~vacant,tree_layout(~Model,left),fill,queue),left,rebuilt(~Model,~Fill,~vacant,fill,left,queued(~Model,right,rest))) :
        {Partition.Fork{_,built(~Model,~Fill,~vacant,tree_layout(~Model,right),fill,taken(~Model,tree_layout(~Model,left),queue))} == Partition.Fork{left,right} : Partition.Tree<&2,Model,Unit>}
      %Equal.sym(List<&2,Model>,taken(~Model,tree_layout(~Model,left),queue),queued(~Model,right,rest),rebuilt_taken(~Model,left,queued(~Model,right,rest))) :
        {Partition.Fork{left,built(~Model,~Fill,~vacant,tree_layout(~Model,right),fill,_)} == Partition.Fork{left,right} : Partition.Tree<&2,Model,Unit>}
      %Equal.sym(Partition.Tree<&2,Model,Unit>,built(~Model,~Fill,~vacant,tree_layout(~Model,right),fill,queued(~Model,right,rest)),right,rebuilt(~Model,~Fill,~vacant,fill,right,rest)) :
        {Partition.Fork{left,_} == Partition.Fork{left,right} : Partition.Tree<&2,Model,Unit>}
      {==}

# A tree regrouped into ranges as many as its blocks and then into its own
# ranges is itself.
def regrouped_twice(~Model: Data,~Fill: Data,~vacant: Fill -> Model,+layout: Pieces.Ranges,+fill: Fill,+tree: Partition.Tree<&2,Model,Unit>,
  +counted: {ranges(layout) == List.length(&2,Model,queued(~Model,tree,[])) : Nat}) ->
  {regrouped(~Model,~Fill,~vacant,tree_layout(~Model,tree),fill,regrouped(~Model,~Fill,~vacant,layout,fill,tree)) == tree : Partition.Tree<&2,Model,Unit>}:
  +queue = queued(~Model,tree,[])
  +zero_counted = Equal.trans(Nat,Nat.add(ranges(layout),0n),ranges(layout),List.length(&2,Model,queue),Addition.zero_right(ranges(layout)),counted)
  +emptied = length_empty(~Model,taken(~Model,layout,queue),taken_count(~Model,layout,queue,0n,zero_counted))
  +kept = {%emptied : {queued(~Model,built(~Model,~Fill,~vacant,layout,fill,queue),_) == queue : List<&2,Model>}
    built_taken(~Model,~Fill,~vacant,layout,fill,queue,0n,zero_counted) : {queued(~Model,built(~Model,~Fill,~vacant,layout,fill,queue),[]) == queue : List<&2,Model>}}
  %Equal.sym(List<&2,Model>,queued(~Model,built(~Model,~Fill,~vacant,layout,fill,queue),[]),queue,kept) :
    {built(~Model,~Fill,~vacant,tree_layout(~Model,tree),fill,_) == tree : Partition.Tree<&2,Model,Unit>}
  rebuilt(~Model,~Fill,~vacant,fill,tree,[])
