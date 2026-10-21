# An indexed store traversal writes sample(reader, context, p) at position p
# for p = 0 .. count - 1. Its output keeps capacity, balance and readiness,
# and inside the written count position p holds its own sample: the job of p
# writes it and no later job writes there. The samples never read the output.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_map_readback.bend as MapReadback
import ./indexed_output_readback.bend as OutputReadback
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./contiguous_copy_readback.bend as Offsets

def output_tree(~Reader: Data,~Element: Data,result: AccessLaws.ModelResult<Reader,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def run(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,position: U32,context: Context,reader: Reader,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  output_tree(~Reader,~Element,AccessLaws.finish_model(~Reader,~Element,~Context,
    Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Reader,~Element,~Context,~sample,~place),
      count,Traversal.State{reader,output,position,context})))

def capacity(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,+position: U32,+context: Context,+reader: Reader,+output: Tree.Tree<Element>) ->
  {Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,count,position,context,reader,output)) == Tree.capacity(Element,output) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = sample(reader,context,position)
      +written = Tree.write(Element,output,place(context,position),value)
      Equal.trans(U32,Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,written)),Tree.capacity(Element,written),Tree.capacity(Element,output),
        capacity(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,written),StorageReadback.write_capacity(~Element,output,place(context,position),value))

def balanced(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,+position: U32,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,output)) -> Shape.Balanced(Element,depth,run(~Reader,~Element,~Context,~sample,~place,count,position,context,reader,output)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = sample(reader,context,position)
      balanced(~Reader,~Element,~Context,~sample,~place,rest,(position + 1 : U32),context,reader,Tree.write(Element,output,place(context,position),value),depth,
        Shape.write(Element,depth,output,place(context,position),value,evidence))

def ready(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  +count: Nat,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+length: Nat,
  evidence: Buffer.ContentReady<Element,Buffer.Content{output,length}>) ->
  Buffer.ContentReady<Element,Buffer.Content{run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),length}>:
  Buffer.ContentReady{+depth,+balance,depth_ok,room} = evidence
  Buffer.ContentReady{depth,balanced(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output,depth,balance),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output)),Tree.capacity(Element,output),
      capacity(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

# One position of the output, asked of the whole traversal.
type Query<-Reader: Data,-Context: Data> is Data:
  Query{count: Nat,position: Nat,depth: Nat,capacity: U32,reader: Reader,context: Context}

def query_count(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  count

def query_position(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  position

def query_depth(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Nat:
  Query{count,position,depth,capacity,reader,context} = query
  depth

def query_capacity(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> U32:
  Query{count,position,depth,capacity,reader,context} = query
  capacity

def query_reader(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Reader:
  Query{count,position,depth,capacity,reader,context} = query
  reader

def query_context(~Reader: Data,~Context: Data,query: Query<Reader,Context>) -> Context:
  Query{count,position,depth,capacity,reader,context} = query
  context

def wanted(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,query: Query<Reader,Context>) -> Element:
  Query{count,position,depth,capacity,reader,context} = query
  sample(reader,context,U32.from_nat(position))

def read(~Reader: Data,~Element: Data,~Context: Data,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Element:
  Tree.read(Element,tree,U32.from_nat(query_position(~Reader,~Context,query)))

def update(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,index: U32,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Query{count,position,depth,capacity,reader,context} = query
  +index = index
  Tree.write(Element,tree,index,sample(reader,context,index))

type Frame<-Reader: Data,-Element: Data,-Context: Data,-query: Query<Reader,Context>,-tree: Tree.Tree<Element>> is Data:
  Frame{capacity_equal: {Tree.capacity(Element,tree) == query_capacity(~Reader,~Context,query) : U32},
    balanced: Shape.Balanced(Element,query_depth(~Reader,~Context,query),tree),
    depth_ok: {Nat.is_le(query_depth(~Reader,~Context,query),30n) == True{} : Bool},
    capacity_ok: {U32.is_le(query_capacity(~Reader,~Context,query),2147483648) == True{} : Bool},
    room: {Nat.is_le(query_count(~Reader,~Context,query),U32.to_nat(query_capacity(~Reader,~Context,query))) == True{} : Bool},
    selected: {Nat.is_lt(query_position(~Reader,~Context,query),query_count(~Reader,~Context,query)) == True{} : Bool}}

def frame(~Reader: Data,~Element: Data,~Context: Data,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Data:
  Frame<Reader,Element,Context,query,tree>

def frame_step(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+index: U32,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>) -> Frame<Reader,Element,Context,query,update(~Reader,~Element,~Context,~sample,index,query,tree)>:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} = evidence
  +value = sample(reader,context,index)
  Frame{Equal.trans(U32,Tree.capacity(Element,Tree.write(Element,tree,index,value)),Tree.capacity(Element,tree),capacity,
      StorageReadback.write_capacity(~Element,tree,index,value),capacity_equal),
    Shape.write(Element,depth,tree,index,value,balanced),depth_ok,capacity_ok,room,selected}

def at_target(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>) ->
  {read(~Reader,~Element,~Context,query,update(~Reader,~Element,~Context,~sample,U32.from_nat(query_position(~Reader,~Context,query)),query,tree)) ==
    wanted(~Reader,~Element,~Context,~sample,query) : Element}:
  Query{count,+position,depth,capacity,+reader,+context} = query
  StorageReadback.read_written(~Element,tree,U32.from_nat(position),sample(reader,context,U32.from_nat(position)))

def later(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,+query: Query<Reader,Context>,+index: Nat,+tree: Tree.Tree<Element>,
  evidence: Frame<Reader,Element,Context,query,tree>,after: {Nat.is_lt(query_position(~Reader,~Context,query),index) == True{} : Bool},
  inside: {Nat.is_lt(index,query_count(~Reader,~Context,query)) == True{} : Bool}) ->
  {read(~Reader,~Element,~Context,query,update(~Reader,~Element,~Context,~sample,U32.from_nat(index),query,tree)) == read(~Reader,~Element,~Context,query,tree) : Element}:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  Frame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,selected} = evidence
  +index_inside = Bounds.nat_trans(index,count,U32.to_nat(capacity),inside,room)
  +position_inside = Bounds.nat_trans(position,count,U32.to_nat(capacity),selected,room)
  +different = {
    %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),U32.from_nat(position)),Nat.is_eq(U32.to_nat(U32.from_nat(index)),U32.to_nat(U32.from_nat(position))),
      Comparison.eq(U32.from_nat(index),U32.from_nat(position))) : {_ == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),index_inside))) :
      {Nat.is_eq(_,U32.to_nat(U32.from_nat(position))) == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
      {Nat.is_eq(index,_) == False{} : Bool}
    %Order.equal_symmetric(position,index) : {_ == False{} : Bool}
    Order.less_not_equal(position,index,after) : {U32.is_eq(U32.from_nat(index),U32.from_nat(position)) == False{} : Bool}}
  MapReadback.unchanged(~Element,depth,tree,U32.from_nat(index),U32.from_nat(position),
    sample(reader,context,U32.from_nat(index)),capacity,capacity_equal,balanced,depth_ok,
    MapReadback.address_inside(index,capacity,capacity_ok,index_inside),MapReadback.address_inside(position,capacity,capacity_ok,position_inside),different)

# The traversal is the indexed fold of its stores once every job writes at
# its own position.
def folded(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~identity: @+context: Context -> @+position: U32 -> {place(context,position) == position : U32},
  +query: Query<Reader,Context>,remaining: Nat,+start: U32,+output: Tree.Tree<Element>) ->
  {run(~Reader,~Element,~Context,~sample,~place,remaining,start,query_context(~Reader,~Context,query),query_reader(~Reader,~Context,query),output) ==
    Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),remaining,start,query,output) : Tree.Tree<Element>}:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  match remaining:
    case 0n: {==}
    case 1n+rest:
      %Equal.sym(U32,place(context,start),start,identity(context,start)) :
        {run(~Reader,~Element,~Context,~sample,~place,rest,(start + 1 : U32),context,reader,Tree.write(Element,output,_,sample(reader,context,start))) ==
          Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),1n+rest,start,
            Query{count,position,depth,capacity,reader,context},output) : Tree.Tree<Element>}
      folded(~Reader,~Element,~Context,~sample,~place,~identity,Query{count,position,depth,capacity,reader,context},rest,(start + 1 : U32),
        Tree.write(Element,output,start,sample(reader,context,start)))

# Inside the written count, a position of the output holds its own sample.
def stored(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~identity: @+context: Context -> @+position: U32 -> {place(context,position) == position : U32},
  +count: Nat,+context: Context,+reader: Reader,+output: Tree.Tree<Element>,+depth: Nat,+capacity: U32,+position: Nat,
  capacity_equal: {Tree.capacity(Element,output) == capacity : U32},balanced: Shape.Balanced(Element,depth,output),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Tree.read(Element,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),U32.from_nat(position)) ==
    sample(reader,context,U32.from_nat(position)) : Element}:
  +query = {Query{count,position,depth,capacity,reader,context} : Query<Reader,Context>}
  %Equal.sym(Tree.Tree<Element>,run(~Reader,~Element,~Context,~sample,~place,count,0,context,reader,output),
    Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~update(~Reader,~Element,~Context,~sample),count,0,query,output),
    folded(~Reader,~Element,~Context,~sample,~place,~identity,query,count,0,output)) :
    {Tree.read(Element,_,U32.from_nat(position)) == sample(reader,context,U32.from_nat(position)) : Element}
  OutputReadback.complete_readback(~Tree.Tree<Element>,~Query<Reader,Context>,~Element,~query_count(~Reader,~Context),~query_position(~Reader,~Context),
    ~wanted(~Reader,~Element,~Context,~sample),~frame(~Reader,~Element,~Context),~read(~Reader,~Element,~Context),~update(~Reader,~Element,~Context,~sample),
    ~frame_step(~Reader,~Element,~Context,~sample),~at_target(~Reader,~Element,~Context,~sample),~later(~Reader,~Element,~Context,~sample),query,output,
    Frame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected},selected)

# An affine destination range may start at any packed base.
type AffineQuery<-Context: Data> is Data:
  AffineQuery{context: Context,base: U32,count: Nat,index: Nat,depth: Nat,capacity: U32}

def affine_count(~Context: Data,query: AffineQuery<Context>) -> Nat:
  AffineQuery{context,base,count,index,depth,capacity} = query
  count

def affine_index(~Context: Data,query: AffineQuery<Context>) -> Nat:
  AffineQuery{context,base,count,index,depth,capacity} = query
  index

def affine_depth(~Context: Data,query: AffineQuery<Context>) -> Nat:
  AffineQuery{context,base,count,index,depth,capacity} = query
  depth

def affine_capacity(~Context: Data,query: AffineQuery<Context>) -> U32:
  AffineQuery{context,base,count,index,depth,capacity} = query
  capacity

def affine_base(~Context: Data,query: AffineQuery<Context>) -> U32:
  AffineQuery{context,base,count,index,depth,capacity} = query
  base

def affine_address(~Context: Data,query: AffineQuery<Context>) -> U32:
  AffineQuery{context,base,count,index,depth,capacity} = query
  (base + U32.from_nat(index) : U32)

def affine_wanted(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,query: AffineQuery<Context>) -> Element:
  AffineQuery{context,base,count,index,depth,capacity} = query
  sample(context,U32.from_nat(index))

def affine_read(~Element: Data,~Context: Data,query: AffineQuery<Context>,tree: Tree.Tree<Element>) -> Element:
  Tree.read(Element,tree,affine_address(~Context,query))

def affine_update(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,position: U32,query: AffineQuery<Context>,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  AffineQuery{context,base,count,index,depth,capacity} = query
  +position = position
  Tree.write(Element,tree,(base + position : U32),sample(context,position))

type AffineFrame<-Element: Data,-Context: Data,-query: AffineQuery<Context>,-tree: Tree.Tree<Element>> is Data:
  AffineFrame{capacity_equal: {Tree.capacity(Element,tree) == affine_capacity(~Context,query) : U32},
    balanced: Shape.Balanced(Element,affine_depth(~Context,query),tree),depth_ok: {Nat.is_le(affine_depth(~Context,query),31n) == True{} : Bool},
    capacity_ok: {U32.is_le(affine_capacity(~Context,query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(affine_base(~Context,query)),affine_count(~Context,query)),U32.to_nat(affine_capacity(~Context,query))) == True{} : Bool},
    selected: {Nat.is_lt(affine_index(~Context,query),affine_count(~Context,query)) == True{} : Bool}}

def affine_frame(~Element: Data,~Context: Data,query: AffineQuery<Context>,tree: Tree.Tree<Element>) -> Data:
  AffineFrame<Element,Context,query,tree>

def affine_frame_step(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,+query: AffineQuery<Context>,+position: U32,+tree: Tree.Tree<Element>,
  evidence: AffineFrame<Element,Context,query,tree>) -> AffineFrame<Element,Context,query,affine_update(~Element,~Context,~sample,position,query,tree)>:
  AffineQuery{+context,+base,count,index,depth,+capacity} = query
  AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected} = evidence
  AffineFrame{Equal.trans(U32,Tree.capacity(Element,affine_update(~Element,~Context,~sample,position,query,tree)),Tree.capacity(Element,tree),capacity,
      StorageReadback.write_capacity(~Element,tree,(base + position : U32),sample(context,position)),capacity_equal),
    Shape.write(Element,depth,tree,(base + position : U32),sample(context,position),balanced),depth_ok,capacity_ok,room,selected}

def affine_at_target(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,+query: AffineQuery<Context>,+tree: Tree.Tree<Element>,
  evidence: AffineFrame<Element,Context,query,tree>) ->
  {affine_read(~Element,~Context,query,affine_update(~Element,~Context,~sample,U32.from_nat(affine_index(~Context,query)),query,tree)) == affine_wanted(~Element,~Context,~sample,query) : Element}:
  AffineQuery{context,+base,count,+index,depth,capacity} = query
  StorageReadback.read_written(~Element,tree,(base + U32.from_nat(index) : U32),sample(context,U32.from_nat(index)))

def affine_later(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,+query: AffineQuery<Context>,+position: Nat,+tree: Tree.Tree<Element>,
  evidence: AffineFrame<Element,Context,query,tree>,after: {Nat.is_lt(affine_index(~Context,query),position) == True{} : Bool},inside: {Nat.is_lt(position,affine_count(~Context,query)) == True{} : Bool}) ->
  {affine_read(~Element,~Context,query,affine_update(~Element,~Context,~sample,U32.from_nat(position),query,tree)) == affine_read(~Element,~Context,query,tree) : Element}:
  AffineQuery{+context,+base,+count,+index,+depth,+capacity} = query
  AffineFrame{+capacity_equal,balanced,depth_ok,+capacity_ok,+room,+selected} = evidence
  +position_inside = Bounds.nat_trans(Nat.add(U32.to_nat(base),position),Nat.add(U32.to_nat(base),count),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(base),position,count,inside),room)
  +index_inside = Bounds.nat_trans(Nat.add(U32.to_nat(base),index),Nat.add(U32.to_nat(base),count),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(base),index,count,selected),room)
  +written = (base + U32.from_nat(position) : U32)
  +observed = (base + U32.from_nat(index) : U32)
  +different = {
    %Equal.sym(Bool,U32.is_eq(written,observed),Nat.is_eq(U32.to_nat(written),U32.to_nat(observed)),Comparison.eq(written,observed)) : {_ == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(written),Nat.add(U32.to_nat(base),position),Offsets.offset_exact(base,position,capacity,capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(base),position),U32.to_nat(capacity),position_inside))) : {Nat.is_eq(_,U32.to_nat(observed)) == False{} : Bool}
    %Equal.sym(Nat,U32.to_nat(observed),Nat.add(U32.to_nat(base),index),Offsets.offset_exact(base,index,capacity,capacity_ok,Bounds.strict_le(Nat.add(U32.to_nat(base),index),U32.to_nat(capacity),index_inside))) : {Nat.is_eq(Nat.add(U32.to_nat(base),position),_) == False{} : Bool}
    %Order.equal_symmetric(Nat.add(U32.to_nat(base),index),Nat.add(U32.to_nat(base),position)) : {_ == False{} : Bool}
    Order.less_not_equal(Nat.add(U32.to_nat(base),index),Nat.add(U32.to_nat(base),position),Order.offset_lt(U32.to_nat(base),index,position,after)) : {U32.is_eq(written,observed) == False{} : Bool}}
  StorageReadback.read_preserved_full(~Element,depth,tree,written,observed,sample(context,U32.from_nat(position)),balanced,depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_equal) : {U32.is_lt(written,_) == True{} : Bool} Offsets.offset_inside(base,position,capacity,capacity_ok,position_inside),
    %Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_equal) : {U32.is_lt(observed,_) == True{} : Bool} Offsets.offset_inside(base,index,capacity,capacity_ok,index_inside),different)

def affine_readback(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,+query: AffineQuery<Context>,+tree: Tree.Tree<Element>,+evidence: AffineFrame<Element,Context,query,tree>) ->
  {affine_read(~Element,~Context,query,Traversal.indexed_fold(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_update(~Element,~Context,~sample),affine_count(~Context,query),0,query,tree)) == affine_wanted(~Element,~Context,~sample,query) : Element}:
  AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,+selected} = evidence
  OutputReadback.complete_readback(~Tree.Tree<Element>,~AffineQuery<Context>,~Element,~affine_count(~Context),~affine_index(~Context),~affine_wanted(~Element,~Context,~sample),~affine_frame(~Element,~Context),
    ~affine_read(~Element,~Context),~affine_update(~Element,~Context,~sample),~affine_frame_step(~Element,~Context,~sample),~affine_at_target(~Element,~Context,~sample),~affine_later(~Element,~Context,~sample),query,tree,evidence,selected)

# A disjoint observation survives a complete affine range of stores.
type AffineKept<-Element: Data,-Context: Data,-observe: Context -> U32,-wanted: AffineQuery<Context> -> Element,-Separated: AffineQuery<Context> -> Data,-query: AffineQuery<Context>,-tree: Tree.Tree<Element>> is Data:
  AffineKept{separated: Separated(query),frame: AffineFrame<Element,Context,query,tree>,
    observed_inside: {U32.is_lt(observe(affine_context(Context,query)),affine_capacity(Context,query)) == True{} : Bool},
    value: {Tree.read(Element,tree,observe(affine_context(Context,query))) == wanted(query) : Element}}

def affine_context(~Context: Data,query: AffineQuery<Context>) -> Context:
  AffineQuery{context,base,count,index,depth,capacity} = query
  context

def affine_kept_step(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,~observe: Context -> U32,~wanted: AffineQuery<Context> -> Element,~Separated: AffineQuery<Context> -> Data,
  ~disjoint: @+query: AffineQuery<Context> -> @+position: Nat -> Separated(query) -> {Nat.is_lt(position,affine_count(~Context,query)) == True{} : Bool} ->
    {U32.is_eq((affine_base(~Context,query) + U32.from_nat(position) : U32),observe(affine_context(~Context,query))) == False{} : Bool},
  +query: AffineQuery<Context>,+position: Nat,+tree: Tree.Tree<Element>,
  inside: {Nat.is_lt(position,affine_count(~Context,query)) == True{} : Bool},
  evidence: AffineKept<Element,Context,observe,wanted,Separated,query,tree>) ->
  AffineKept<Element,Context,observe,wanted,Separated,query,affine_update(~Element,~Context,~sample,U32.from_nat(position),query,tree)>:
  AffineQuery{+context,+base,+count,index,+depth,+capacity} = query
  AffineKept{+separated,+frame,+observed_inside,value} = evidence
  AffineFrame{+capacity_equal,balanced,depth_ok,+capacity_ok,room,selected} = frame
  +inside = inside
  +written = (base + U32.from_nat(position) : U32)
  +observed = observe(context)
  AffineKept{separated,affine_frame_step(~Element,~Context,~sample,query,U32.from_nat(position),tree,frame),observed_inside,
    Equal.trans(Element,Tree.read(Element,affine_update(~Element,~Context,~sample,U32.from_nat(position),query,tree),observed),Tree.read(Element,tree,observed),wanted(query),
      StorageReadback.read_preserved_full(~Element,depth,tree,written,observed,sample(context,U32.from_nat(position)),balanced,depth_ok,
        %Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_equal) : {U32.is_lt(written,_) == True{} : Bool}
          Offsets.offset_inside(base,position,capacity,capacity_ok,
            Bounds.nat_trans(Nat.add(U32.to_nat(base),position),Nat.add(U32.to_nat(base),count),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(base),position,count,inside),room)),
        %Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_equal) : {U32.is_lt(observed,_) == True{} : Bool} observed_inside,
        disjoint(query,position,separated,inside)),value)}

def affine_kept_value(~Element: Data,~Context: Data,~observe: Context -> U32,~wanted: AffineQuery<Context> -> Element,~Separated: AffineQuery<Context> -> Data,
  query: AffineQuery<Context>,tree: Tree.Tree<Element>,evidence: AffineKept<Element,Context,observe,wanted,Separated,query,tree>) ->
  {Tree.read(Element,tree,observe(affine_context(~Context,query))) == wanted(query) : Element}:
  AffineKept{separated,frame,observed_inside,value} = evidence
  value

def affine_preserves(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,~observe: Context -> U32,~wanted: AffineQuery<Context> -> Element,~Separated: AffineQuery<Context> -> Data,
  ~disjoint: @+query: AffineQuery<Context> -> @+position: Nat -> Separated(query) -> {Nat.is_lt(position,affine_count(~Context,query)) == True{} : Bool} ->
    {U32.is_eq((affine_base(~Context,query) + U32.from_nat(position) : U32),observe(affine_context(~Context,query))) == False{} : Bool},
  +query: AffineQuery<Context>,+tree: Tree.Tree<Element>,+frame: AffineFrame<Element,Context,query,tree>,separated: Separated(query),
  observed_inside: {U32.is_lt(observe(affine_context(~Context,query)),affine_capacity(~Context,query)) == True{} : Bool},
  initial: {Tree.read(Element,tree,observe(affine_context(~Context,query))) == wanted(query) : Element}) ->
  {Tree.read(Element,Traversal.indexed_fold(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_update(~Element,~Context,~sample),affine_count(~Context,query),0,query,tree),
    observe(affine_context(~Context,query))) == wanted(query) : Element}:
  affine_kept_value(~Element,~Context,~observe,~wanted,~Separated,query,
    Traversal.indexed_fold(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_update(~Element,~Context,~sample),affine_count(~Context,query),0,query,tree),
    ArrayLaws.indexed_fold_preserves_invariant(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_count(~Context),
      ~(query => index => tree => AffineKept<Element,Context,observe,wanted,Separated,query,tree>),~affine_update(~Element,~Context,~sample),
      ~affine_kept_step(~Element,~Context,~sample,~observe,~wanted,~Separated,~disjoint),affine_count(~Context,query),0n,query,tree,
      Order.reflexive(affine_count(~Context,query)),AffineKept{separated,frame,observed_inside,initial}))

type AffineBefore<-Context: Data,-observe: Context -> U32,-query: AffineQuery<Context>> is Data:
  AffineBefore{capacity_ok: {U32.is_le(affine_capacity(Context,query),2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(affine_base(Context,query)),affine_count(Context,query)),U32.to_nat(affine_capacity(Context,query))) == True{} : Bool},
    before: {Nat.is_lt(U32.to_nat(observe(affine_context(Context,query))),U32.to_nat(affine_base(Context,query))) == True{} : Bool}}

def affine_disjoint_before(~Context: Data,~observe: Context -> U32,+query: AffineQuery<Context>,+position: Nat,
  evidence: AffineBefore<Context,observe,query>,inside: {Nat.is_lt(position,affine_count(~Context,query)) == True{} : Bool}) ->
  {U32.is_eq((affine_base(~Context,query) + U32.from_nat(position) : U32),observe(affine_context(~Context,query))) == False{} : Bool}:
  AffineQuery{+context,+base,+count,index,depth,+capacity} = query
  AffineBefore{capacity_ok,room,before} = evidence
  +written = (base + U32.from_nat(position) : U32)
  +observed = observe(context)
  %Equal.sym(Bool,U32.is_eq(written,observed),Nat.is_eq(U32.to_nat(written),U32.to_nat(observed)),Comparison.eq(written,observed)) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(written),Nat.add(U32.to_nat(base),position),Offsets.offset_exact(base,position,capacity,capacity_ok,
    Bounds.strict_le(Nat.add(U32.to_nat(base),position),U32.to_nat(capacity),
      Bounds.nat_trans(Nat.add(U32.to_nat(base),position),Nat.add(U32.to_nat(base),count),U32.to_nat(capacity),Order.offset_lt(U32.to_nat(base),position,count,inside),room)))) :
    {Nat.is_eq(_,U32.to_nat(observed)) == False{} : Bool}
  %Order.equal_symmetric(U32.to_nat(observed),Nat.add(U32.to_nat(base),position)) : {_ == False{} : Bool}
  Order.less_not_equal(U32.to_nat(observed),Nat.add(U32.to_nat(base),position),
    Bounds.nat_trans(U32.to_nat(observed),U32.to_nat(base),Nat.add(U32.to_nat(base),position),before,Order.le_add(U32.to_nat(base),position)))

def affine_observed_before(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,~observe: Context -> U32,~wanted: AffineQuery<Context> -> Element,
  +query: AffineQuery<Context>,+tree: Tree.Tree<Element>,+frame: AffineFrame<Element,Context,query,tree>,
  evidence: AffineBefore<Context,observe,query>,observed_inside: {U32.is_lt(observe(affine_context(~Context,query)),affine_capacity(~Context,query)) == True{} : Bool},
  initial: {Tree.read(Element,tree,observe(affine_context(~Context,query))) == wanted(query) : Element}) ->
  {Tree.read(Element,Traversal.indexed_fold(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_update(~Element,~Context,~sample),affine_count(~Context,query),0,query,tree),
    observe(affine_context(~Context,query))) == wanted(query) : Element}:
  affine_preserves(~Element,~Context,~sample,~observe,~wanted,~(query => AffineBefore<Context,observe,query>),~affine_disjoint_before(~Context,~observe),query,tree,frame,evidence,observed_inside,initial)

# The observation is runtime data in the context, never a captured template.
type Observed<-Element: Data,-Context: Data> is Data:
  Observed{context: Context,address: U32,original: Tree.Tree<Element>}

def observed_sample(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,context: Observed<Element,Context>,index: U32) -> Element:
  Observed{context,address,original} = context
  sample(context,index)

def observed_address(~Element: Data,~Context: Data,context: Observed<Element,Context>) -> U32:
  Observed{context,address,original} = context
  address

def observed_query(~Element: Data,~Context: Data,query: AffineQuery<Context>,address: U32,original: Tree.Tree<Element>) -> AffineQuery<Observed<Element,Context>>:
  AffineQuery{context,base,count,index,depth,capacity} = query
  AffineQuery{Observed{context,address,original},base,count,index,depth,capacity}

def original_query(~Element: Data,~Context: Data,query: AffineQuery<Observed<Element,Context>>) -> AffineQuery<Context>:
  AffineQuery{Observed{context,address,original},base,count,index,depth,capacity} = query
  AffineQuery{context,base,count,index,depth,capacity}

def observed_wanted(~Element: Data,~Context: Data,query: AffineQuery<Observed<Element,Context>>) -> Element:
  AffineQuery{Observed{context,address,original},base,count,index,depth,capacity} = query
  Tree.read(Element,original,address)

def observed_step(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,+position: U32,+query: AffineQuery<Observed<Element,Context>>,+tree: Tree.Tree<Element>) ->
  {affine_update(~Element,~Observed<Element,Context>,~observed_sample(~Element,~Context,~sample),position,query,tree) ==
    affine_update(~Element,~Context,~sample,position,original_query(~Element,~Context,query),tree) : Tree.Tree<Element>}:
  AffineQuery{Observed{context,address,original},base,count,index,depth,capacity} = query
  {==}

def affine_read_before(~Element: Data,~Context: Data,~sample: Context -> U32 -> Element,
  +query: AffineQuery<Context>,+address: U32,+tree: Tree.Tree<Element>,
  frame: AffineFrame<Element,Context,query,tree>,
  before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(affine_base(~Context,query))) == True{} : Bool},
  inside: {U32.is_lt(address,affine_capacity(~Context,query)) == True{} : Bool}) ->
  {Tree.read(Element,Traversal.indexed_fold(~Tree.Tree<Element>,~AffineQuery<Context>,~affine_update(~Element,~Context,~sample),affine_count(~Context,query),0,query,tree),address) == Tree.read(Element,tree,address) : Element}:
  AffineQuery{context,base,+count,index,depth,capacity} = query
  AffineFrame{capacity_equal,balanced,depth_ok,+capacity_ok,+room,selected} = frame
  +observed = observed_query(~Element,~Context,query,address,tree)
  %ArrayLaws.indexed_fold_change_context(~Tree.Tree<Element>,~AffineQuery<Observed<Element,Context>>,~AffineQuery<Context>,~original_query(~Element,~Context),
    ~affine_update(~Element,~Observed<Element,Context>,~observed_sample(~Element,~Context,~sample)),~affine_update(~Element,~Context,~sample),~observed_step(~Element,~Context,~sample),count,0,observed,tree) :
    {Tree.read(Element,_,address) == Tree.read(Element,tree,address) : Element}
  affine_observed_before(~Element,~Observed<Element,Context>,~observed_sample(~Element,~Context,~sample),~observed_address(~Element,~Context),~observed_wanted(~Element,~Context),observed,tree,
    AffineFrame{capacity_equal,balanced,depth_ok,capacity_ok,room,selected},AffineBefore{capacity_ok,room,before},inside,{==})

# The existing traversal shape proofs also apply to an indexed fold at any destination.
def placed_update(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  position: U32,query: Query<Reader,Context>,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Query{count,index,depth,capacity,reader,+context} = query
  +position = position
  Tree.write(Element,tree,place(context,position),sample(reader,context,position))

def placed_folded(~Reader: Data,~Element: Data,~Context: Data,~sample: Reader -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  +query: Query<Reader,Context>,remaining: Nat,+start: U32,+output: Tree.Tree<Element>) ->
  {run(~Reader,~Element,~Context,~sample,~place,remaining,start,query_context(~Reader,~Context,query),query_reader(~Reader,~Context,query),output) ==
    Traversal.indexed_fold(~Tree.Tree<Element>,~Query<Reader,Context>,~placed_update(~Reader,~Element,~Context,~sample,~place),remaining,start,query,output) : Tree.Tree<Element>}:
  Query{+count,+position,+depth,+capacity,+reader,+context} = query
  match remaining:
    case 0n: {==}
    case 1n+rest:
      placed_folded(~Reader,~Element,~Context,~sample,~place,query,rest,(start + 1 : U32),Tree.write(Element,output,place(context,start),sample(reader,context,start)))
