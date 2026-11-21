# A traversal in blocks computes the flat model traversal whose writes agree
# with each block's own inside it. Block b holds the positions b * size to
# b * size + size - 1, every block lies in total, and total in the machine.
import Base
import ../machine_limits.bend as Machine
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_blocks.bend as Blocks
import ./storage_tree_model.bend as Tree
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./loop_projection.bend as Projection
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition

# ---- The blocks as models ----

def model_next(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,block: U32,size: U32,context: Context,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>:
  Traversal.State{input,output,position,inner} = state
  Traversal.State{input,output,position,Blocks.Blocks{(block + 1 : U32),size,context}}

def model_block_step(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>:
  Traversal.State{input,output,+position,Blocks.Blocks{+block,+size,+context}} = state
  model_next(~Model,~Element,~Context,~Inner,block,size,context,
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample,~place),U32.to_nat(size),
      Traversal.State{input,output,position,inner(context,block,position)}))

def next_represented(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~pack_reader: Model -> Reader,+block: U32,+size: U32,+context: Context,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>) ->
  {Blocks.next_block(~Reader,~Element,~Context,~Inner,block,size,context,ArrayLaws.represent(~Reader,~Model,~Element,~Inner,~pack_reader,state)) ==
    ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,model_next(~Model,~Element,~Context,~Inner,block,size,context,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>}:
  Traversal.State{input,output,position,inner} = state
  {==}

def block_step_refines(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~pack_reader: Model -> Reader,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Inner -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) ->
  {Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place,ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,state)) ==
    ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>}:
  Traversal.State{+input,+output,+position,Blocks.Blocks{+block,+size,+context}} = state
  %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Inner>,
    Traversal.run(~Reader,~Element,~Inner,~read,~place,U32.to_nat(size),
      ArrayLaws.represent(~Reader,~Model,~Element,~Inner,~pack_reader,Traversal.State{input,output,position,inner(context,block,position)})),
    ArrayLaws.represent(~Reader,~Model,~Element,~Inner,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample,~place),U32.to_nat(size),
        Traversal.State{input,output,position,inner(context,block,position)})),
    ArrayLaws.traversal_refines_model(~Reader,~Model,~Element,~Inner,~pack_reader,~read,~sample,~place,~read_matches,U32.to_nat(size),
      Traversal.State{input,output,position,inner(context,block,position)})) :
    {Blocks.next_block(~Reader,~Element,~Context,~Inner,block,size,context,_) ==
      ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,
        model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place,Traversal.State{input,output,position,Blocks.Blocks{block,size,context}})) :
      Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>}
  next_represented(~Reader,~Model,~Element,~Context,~Inner,~pack_reader,block,size,context,
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample,~place),U32.to_nat(size),
      Traversal.State{input,output,position,inner(context,block,position)}))

# ---- One block is a stretch of the flat traversal ----

# The state with the flat context in place of a block's.
def within(~Model: Data,~Element: Data,~Inner: Data,~Context: Data,context: Context,state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>) ->
  Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{input,output,position,inner} = state
  Traversal.State{input,output,position,context}

def flatten(~Model: Data,~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) ->
  Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{input,output,position,Blocks.Blocks{block,size,context}} = state
  Traversal.State{input,output,position,context}

def output_of(~Model: Data,~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Tree.Tree<Element>:
  Traversal.State{input,output,position,context} = state
  output

# The next block depends only on where the last one left the reader, the
# output and the position.
def next_within(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,+block: U32,+size: U32,+context: Context,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>) ->
  {model_next(~Model,~Element,~Context,~Inner,block,size,context,state) ==
    model_next(~Model,~Element,~Context,~Context,block,size,context,within(~Model,~Element,~Inner,~Context,context,state)) :
    Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>}:
  Traversal.State{input,output,position,inner} = state
  {==}

# count steps from position start end at start + count.
def advance(~Model: Data,~Element: Data,~Context: Data,~sample: Model -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  count: Nat,+start: Nat,+reader: Model,+output: Tree.Tree<Element>,+context: Context) ->
  {Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),count,
      Traversal.State{reader,output,U32.from_nat(start),context}) ==
    Traversal.State{reader,
      output_of(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),count,
        Traversal.State{reader,output,U32.from_nat(start),context})),
      U32.from_nat(Nat.add(start,count)),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.State{reader,output,U32.from_nat(start),context} == Traversal.State{reader,output,U32.from_nat(_),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      {==}
    case 1n+ +rest:
      +position = U32.from_nat(start)
      +written = Tree.write(Element,output,place(context,position),sample(reader,context,position))
      %Equal.sym(U32,(position + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(position)) :
        {Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
            Traversal.State{reader,written,_,context}) ==
          Traversal.State{reader,
            output_of(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
              Traversal.State{reader,written,_,context})),
            U32.from_nat(Nat.add(start,1n+rest)),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
            Traversal.State{reader,written,U32.from_nat(1n+start),context}) ==
          Traversal.State{reader,
            output_of(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
              Traversal.State{reader,written,U32.from_nat(1n+start),context})),
            U32.from_nat(_),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      advance(~Model,~Element,~Context,~sample,~place,rest,1n+start,reader,written,context)

# Inside block block the block's writes are the flat writes.
def segment(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample_in: Model -> Inner -> U32 -> Element,~place_in: Inner -> U32 -> U32,~size: Context -> U32,
  ~sample: Model -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~same: @+reader: Model -> @+output: Tree.Tree<Element> -> @+context: Context -> @+limits: Machine.Limits -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool} ->
    {Tree.write(Element,output,
        place_in(inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample_in(reader,inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) ==
      Tree.write(Element,output,place(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample(reader,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) : Tree.Tree<Element>},
  count: Nat,+offset: Nat,+block: Nat,+limits: Machine.Limits,+total: U32,+context: Context,+reader: Model,+output: Tree.Tree<Element>,
  fits: {Nat.is_le(Nat.add(offset,count),U32.to_nat(size(context))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool},
  +bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {within(~Model,~Element,~Inner,~Context,context,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),count,
        Traversal.State{reader,output,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),
          inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))))})) ==
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),count,
      Traversal.State{reader,output,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)),context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +fits = fits
      +s = U32.to_nat(size(context))
      +first = Nat.mul(block,s)
      +position = U32.from_nat(Nat.add(first,offset))
      +local = inner(context,U32.from_nat(block),U32.from_nat(first))
      +written = Tree.write(Element,output,place(context,position),sample(reader,context,position))
      +next = U32.from_nat(Nat.add(first,1n+offset))
      +moved = Equal.trans(U32,(position + 1 : U32),U32.from_nat(1n+Nat.add(first,offset)),next,Bounds.add_one(position),
        Equal.cong(Nat,U32,value => U32.from_nat(value),1n+Nat.add(first,offset),Nat.add(first,1n+offset),
          Equal.sym(Nat,Nat.add(first,1n+offset),1n+Nat.add(first,offset),Addition.successor_right(first,offset))))
      %Equal.sym(Tree.Tree<Element>,Tree.write(Element,output,place_in(local,position),sample_in(reader,local,position)),written,
        same(reader,output,context,limits,total,block,offset,Order.room_index(offset,rest,s,fits),room,bounded)) :
        {within(~Model,~Element,~Inner,~Context,context,
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),rest,
              Traversal.State{reader,_,(position + 1 : U32),local})) ==
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
            Traversal.State{reader,written,(position + 1 : U32),context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      %Equal.sym(U32,(position + 1 : U32),next,moved) :
        {within(~Model,~Element,~Inner,~Context,context,
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),rest,
              Traversal.State{reader,written,_,local})) ==
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),rest,
            Traversal.State{reader,written,_,context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      segment(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in,~size,~sample,~place,~same,rest,1n+offset,block,limits,total,context,reader,written,
        Order.advance_room(offset,rest,s,fits),room,bounded)

# ---- count blocks are count * size flat steps ----

def block_room(+first: Nat,+size: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(first,Nat.add(size,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(first,size),limit) == True{} : Bool}:
  Order.transitive(Nat.add(first,size),Nat.add(Nat.add(first,size),rest),limit,Order.le_add(Nat.add(first,size),rest),
    %Equal.sym(Nat,Nat.add(Nat.add(first,size),rest),Nat.add(first,Nat.add(size,rest)),Addition.associative(first,size,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
    room)

def next_room(+first: Nat,+size: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.add(first,Nat.add(size,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(size,first),rest),limit) == True{} : Bool}:
  %Order.commutative(first,size) : {Nat.is_le(Nat.add(_,rest),limit) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(first,size),rest),Nat.add(first,Nat.add(size,rest)),Addition.associative(first,size,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

def blocks_flat(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample_in: Model -> Inner -> U32 -> Element,~place_in: Inner -> U32 -> U32,~size: Context -> U32,
  ~sample: Model -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~same: @+reader: Model -> @+output: Tree.Tree<Element> -> @+context: Context -> @+limits: Machine.Limits -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool} ->
    {Tree.write(Element,output,
        place_in(inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample_in(reader,inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) ==
      Tree.write(Element,output,place(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample(reader,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) : Tree.Tree<Element>},
  count: Nat,+block: Nat,+limits: Machine.Limits,+total: U32,+context: Context,+reader: Model,+output: Tree.Tree<Element>,
  +room: {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),Nat.mul(count,U32.to_nat(size(context)))),U32.to_nat(total)) == True{} : Bool},
  +bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {flatten(~Model,~Element,~Context,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in),count,
        Traversal.State{reader,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),Blocks.Blocks{U32.from_nat(block),size(context),context}})) ==
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(count,U32.to_nat(size(context))),
      Traversal.State{reader,output,U32.from_nat(Nat.mul(block,U32.to_nat(size(context)))),context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +s = U32.to_nat(size(context))
      +first = Nat.mul(block,s)
      +start = {Traversal.State{reader,output,U32.from_nat(first),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      +local = inner(context,U32.from_nat(block),U32.from_nat(first))
      +done = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),s,start)
      +written = output_of(~Model,~Element,~Context,done)
      +ended = {Traversal.State{reader,written,U32.from_nat(Nat.mul(1n+block,s)),context} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      +moved = Equal.trans(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,done,Traversal.State{reader,written,U32.from_nat(Nat.add(first,s)),context},ended,
        advance(~Model,~Element,~Context,~sample,~place,s,first,reader,output,context),
        Equal.cong(Nat,Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,value => Traversal.State{reader,written,U32.from_nat(value),context},Nat.add(first,s),Nat.add(s,first),
          Order.commutative(first,s)))
      +own = {
        %Addition.zero_right(first) :
          {within(~Model,~Element,~Inner,~Context,context,
              Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),s,
                Traversal.State{reader,output,U32.from_nat(_),local})) ==
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),s,
              Traversal.State{reader,output,U32.from_nat(_),context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
        segment(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in,~size,~sample,~place,~same,s,0n,block,limits,total,context,reader,output,
          Order.reflexive(s),block_room(first,s,Nat.mul(rest,s),U32.to_nat(total),room),bounded) :
        {within(~Model,~Element,~Inner,~Context,context,
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),s,
              Traversal.State{reader,output,U32.from_nat(first),local})) == done : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}}
      +inside = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample_in,~place_in),s,
        Traversal.State{reader,output,U32.from_nat(first),local})
      +stepped = {
        %Bounds.add_one(U32.from_nat(block)) :
          {model_next(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside) ==
            Traversal.State{reader,written,U32.from_nat(Nat.mul(1n+block,s)),Blocks.Blocks{_,size(context),context}} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>}
        Equal.trans(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,
          model_next(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside),
          model_next(~Model,~Element,~Context,~Context,U32.from_nat(block),size(context),context,within(~Model,~Element,~Inner,~Context,context,inside)),
          model_next(~Model,~Element,~Context,~Context,U32.from_nat(block),size(context),context,ended),
          next_within(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside),
          Equal.cong(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,
            state => model_next(~Model,~Element,~Context,~Context,U32.from_nat(block),size(context),context,state),
            within(~Model,~Element,~Inner,~Context,context,inside),ended,
            Equal.trans(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,within(~Model,~Element,~Inner,~Context,context,inside),done,ended,own,moved))) :
        {model_next(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside) ==
          Traversal.State{reader,written,U32.from_nat(Nat.mul(1n+block,s)),Blocks.Blocks{U32.from_nat(1n+block),size(context),context}} :
          Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>}}
      +later = blocks_flat(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in,~size,~sample,~place,~same,rest,1n+block,limits,total,context,reader,written,
        next_room(first,s,Nat.mul(rest,s),U32.to_nat(total),room),bounded)
      Equal.trans(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
        flatten(~Model,~Element,~Context,
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in),rest,
            model_next(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside))),
        Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),ended),
        Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.add(s,Nat.mul(rest,s)),start),
        %Equal.sym(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,model_next(~Model,~Element,~Context,~Inner,U32.from_nat(block),size(context),context,inside),
            Traversal.State{reader,written,U32.from_nat(Nat.mul(1n+block,s)),Blocks.Blocks{U32.from_nat(1n+block),size(context),context}},stepped) :
          {flatten(~Model,~Element,~Context,
              Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in),rest,_)) ==
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),ended) :
            Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
        later,
        Equal.sym(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.add(s,Nat.mul(rest,s)),start),
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),ended),
          Equal.trans(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.add(s,Nat.mul(rest,s)),start),
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),done),
            Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),ended),
            AccessLaws.run_added(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),s,Nat.mul(rest,s),start),
            Equal.cong(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
              state => Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),Nat.mul(rest,s),state),
              done,ended,moved))))

# ---- The run ----

# The model of the run: blocks block steps from position 0.
def blocked(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,blocks: Nat,size: U32,context: Context,input: Model,output: Tree.Tree<Element>) ->
  AccessLaws.ModelResult<Model,Element>:
  AccessLaws.finish_model(~Model,~Element,~Blocks.Blocks<Context>,
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),blocks,
      Traversal.State{input,output,0,Blocks.Blocks{0,size,context}}))

# The run is its model, for every count and size.
def represented(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~pack_reader: Model -> Reader,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Inner -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +blocks: Nat,+size: U32,+context: Context,+input: Model,+output: Tree.Tree<Element>) ->
  {Blocks.run(~Reader,~Element,~Context,~Inner,~inner,~read,~place,blocks,size,context,pack_reader(input),Tree.pack(Element,output)) ==
    AccessLaws.pack_result(~Reader,~Model,~Element,~pack_reader,blocked(~Model,~Element,~Context,~Inner,~inner,~sample,~place,blocks,size,context,input,output)) :
    Reader & Array<Element>}:
  +start = {Traversal.State{input,output,0,Blocks.Blocks{0,size,context}} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>}
  +done = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),blocks,start)
  %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,
    Loop.run(~Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,~Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place),blocks,
      ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,start)),
    ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,done),
    Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,
      ~Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place),~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),
      ~ArrayLaws.represent(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader),
      ~block_step_refines(~Reader,~Model,~Element,~Context,~Inner,~pack_reader,~inner,~read,~sample,~place,~read_matches),blocks,start)) :
    {Traversal.finish(~Reader,~Element,~Blocks.Blocks<Context>,_) ==
      AccessLaws.pack_result(~Reader,~Model,~Element,~pack_reader,AccessLaws.finish_model(~Model,~Element,~Blocks.Blocks<Context>,done)) : Reader & Array<Element>}
  AccessLaws.indexed_finished(~Reader,~Model,~Element,~Blocks.Blocks<Context>,~pack_reader,done)

def flat_finish(~Model: Data,~Element: Data,~Context: Data,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) ->
  {AccessLaws.finish_model(~Model,~Element,~Blocks.Blocks<Context>,state) == AccessLaws.finish_model(~Model,~Element,~Context,flatten(~Model,~Element,~Context,state)) :
    AccessLaws.ModelResult<Model,Element>}:
  Traversal.State{input,output,position,Blocks.Blocks{block,size,context}} = state
  {==}

def whole_room(+product: Nat,+limit: Nat,exact: {product == limit : Nat}) -> {Nat.is_le(product,limit) == True{} : Bool}:
  %exact : {Nat.is_le(product,_) == True{} : Bool}
  Order.reflexive(product)

# blocks blocks of size(context) positions that are the total positions within
# the machine leave the result of the flat model traversal of total steps.
def flat(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample_in: Model -> Inner -> U32 -> Element,~place_in: Inner -> U32 -> U32,~size: Context -> U32,
  ~sample: Model -> Context -> U32 -> Element,~place: Context -> U32 -> U32,
  ~same: @+reader: Model -> @+output: Tree.Tree<Element> -> @+context: Context -> @+limits: Machine.Limits -> @+total: U32 -> @+block: Nat -> @+offset: Nat ->
    {Nat.is_lt(offset,U32.to_nat(size(context))) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(block,U32.to_nat(size(context))),U32.to_nat(size(context))),U32.to_nat(total)) == True{} : Bool} ->
    {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool} ->
    {Tree.write(Element,output,
        place_in(inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample_in(reader,inner(context,U32.from_nat(block),U32.from_nat(Nat.mul(block,U32.to_nat(size(context))))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) ==
      Tree.write(Element,output,place(context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset))),
        sample(reader,context,U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(size(context))),offset)))) : Tree.Tree<Element>},
  +blocks: Nat,+limits: Machine.Limits,+total: U32,+context: Context,+input: Model,+output: Tree.Tree<Element>,
  +exact: {Nat.mul(blocks,U32.to_nat(size(context))) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool}) ->
  {blocked(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in,blocks,size(context),context,input,output) ==
    AccessLaws.finish_model(~Model,~Element,~Context,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),U32.to_nat(total),
        Traversal.State{input,output,0,context})) : AccessLaws.ModelResult<Model,Element>}:
  +done = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in),blocks,
    Traversal.State{input,output,0,Blocks.Blocks{0,size(context),context}})
  +whole = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),U32.to_nat(total),
    Traversal.State{input,output,0,context})
  Equal.trans(AccessLaws.ModelResult<Model,Element>,AccessLaws.finish_model(~Model,~Element,~Blocks.Blocks<Context>,done),
    AccessLaws.finish_model(~Model,~Element,~Context,flatten(~Model,~Element,~Context,done)),AccessLaws.finish_model(~Model,~Element,~Context,whole),
    flat_finish(~Model,~Element,~Context,done),
    Equal.cong(Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,AccessLaws.ModelResult<Model,Element>,state => AccessLaws.finish_model(~Model,~Element,~Context,state),
      flatten(~Model,~Element,~Context,done),whole,
      %exact : {flatten(~Model,~Element,~Context,done) ==
          Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~place),_,
            Traversal.State{input,output,0,context}) : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}
      blocks_flat(~Model,~Element,~Context,~Inner,~inner,~sample_in,~place_in,~size,~sample,~place,~same,blocks,0n,limits,total,context,input,output,
        whole_room(Nat.mul(blocks,U32.to_nat(size(context))),U32.to_nat(total),exact),bounded)))


# ---- The reader ----

def next_reader(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,+block: U32,+size: U32,+context: Context,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>) ->
  {ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,model_next(~Model,~Element,~Context,~Inner,block,size,context,state)) ==
    ArrayLaws.model_reader(~Model,~Element,~Inner,state) : Model}:
  Traversal.State{input,output,position,inner} = state
  {==}

# A block step and a run of blocks keep the reader.
def block_reader_step(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) ->
  {ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place,state)) ==
    ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,state) : Model}:
  Traversal.State{+input,+output,+position,Blocks.Blocks{+block,+size,+context}} = state
  +start = {Traversal.State{input,output,position,inner(context,block,position)} : Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>}
  +done = Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Inner>,~ArrayLaws.model_step(~Model,~Element,~Inner,~sample,~place),U32.to_nat(size),start)
  Equal.trans(Model,ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,model_next(~Model,~Element,~Context,~Inner,block,size,context,done)),
    ArrayLaws.model_reader(~Model,~Element,~Inner,done),input,next_reader(~Model,~Element,~Context,~Inner,block,size,context,done),
    ArrayLaws.model_reader_kept(~Model,~Element,~Inner,~sample,~place,U32.to_nat(size),start))

def blocks_reader_kept(~Model: Data,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~sample: Model -> Inner -> U32 -> Element,~place: Inner -> U32 -> U32,+count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>) ->
  {ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),count,state)) ==
    ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>,state) : Model}:
  Projection.preserves_observation(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Blocks.Blocks<Context>>,~Model,
    ~model_block_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),~ArrayLaws.model_reader(~Model,~Element,~Blocks.Blocks<Context>),
    ~block_reader_step(~Model,~Element,~Context,~Inner,~inner,~sample,~place),count,state)
