# The value an ordered in-place update loop leaves at one address: the
# positions in order, each combining the value at the address with its sample
# when it writes there. A run that stores its samples is the update that keeps
# the sample.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../kernel_gemm.bend as KernelGemm
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./write_program_readback.bend as WriteReadback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

def state_tree(~Model: Data,~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Tree.Tree<Element>:
  Traversal.State{reader,output,position,context} = state
  output

# The value at query after count positions from position on.
def updated_at(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+query: U32,+current: Element) -> Element:
  match count:
    case 0n: current
    case 1n+rest:
      updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,
        Bool.pick(Element,U32.is_eq(destination(context,position),query),combine(current,sample(reader,context,position)),current))

# Every destination of count positions from position on lies below capacity.
def Destinations(~Context: Data,~destination: Context -> U32 -> U32,count: Nat,+position: U32,+context: Context,+capacity: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<{U32.is_lt(destination(context,position),capacity) == True{} : Bool},Destinations(~Context,~destination,rest,(position + 1 : U32),context,capacity)>

def chosen(~Element: Data,~Sample: Data,~combine: Element -> Sample -> Element,+tree: Tree.Tree<Element>,+address: U32,+query: U32,+sample: Sample,equal: Bool,
  +compared: {U32.is_eq(address,query) == equal : Bool}) ->
  {Bool.pick(Element,equal,combine(Tree.read(Element,tree,address),sample),Tree.read(Element,tree,query)) ==
    Bool.pick(Element,equal,combine(Tree.read(Element,tree,query),sample),Tree.read(Element,tree,query)) : Element}:
  match equal:
    case True{}:
      %Arithmetic.equal_of_is_eq(address,query,compared) : {combine(Tree.read(Element,tree,address),sample) == combine(Tree.read(Element,tree,_),sample) : Element}
      {==}
    case False{}: {==}

# The loop leaves at query the value the positions make of the value there.
def update_read(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+tree: Tree.Tree<Element>,+depth: Nat,+query: U32,
  +balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool},+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  inside: Destinations(~Context,~destination,count,position,context,Tree.capacity(Element,tree))) ->
  {Tree.read(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
      ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,position,context})),query) ==
    updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,position,reader,context,query,Tree.read(Element,tree,query)) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+address_inside,rest_inside} = inside
      +address = destination(context,position)
      +taken = sample(reader,context,position)
      +value = combine(Tree.read(Element,tree,address),taken)
      +written = Tree.write(Element,tree,address,value)
      +capacity_equal = StorageReadback.write_capacity(~Element,tree,address,value)
      +picked = Bool.pick(Element,U32.is_eq(address,query),combine(Tree.read(Element,tree,query),taken),Tree.read(Element,tree,query))
      Equal.trans(Element,
        Tree.read(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
          ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),rest,Traversal.State{reader,written,(position + 1 : U32),context})),query),
        updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,Tree.read(Element,written,query)),
        updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,picked),
        update_read(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,written,depth,query,
          Shape.write(Element,depth,tree,address,value,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : Destinations(~Context,~destination,rest,(position + 1 : U32),context,_) rest_inside),
        Equal.cong(Element,Element,current => updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,current),
          Tree.read(Element,written,query),picked,
          Equal.trans(Element,Tree.read(Element,written,query),Bool.pick(Element,U32.is_eq(address,query),value,Tree.read(Element,tree,query)),picked,
            WriteReadback.one_full(~Element,depth,tree,address,query,value,balanced,limit,address_inside,query_inside),
            chosen(~Element,~Sample,~combine,tree,address,query,taken,U32.is_eq(address,query),{==}))))

# ---- Positions that write the address or not ----

# No position of count from position on writes query.
def Misses(~Context: Data,~destination: Context -> U32 -> U32,count: Nat,+position: U32,+context: Context,+query: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<{U32.is_eq(destination(context,position),query) == False{} : Bool},Misses(~Context,~destination,rest,(position + 1 : U32),context,query)>

# Only the position before positions after position, at, writes query.
def Hits(~Context: Data,~destination: Context -> U32 -> U32,count: Nat,+position: U32,+context: Context,+query: U32,+at: U32,before: Nat) -> Data:
  match count:
    case 0n: Empty
    case 1n+rest:
      match before:
        case 0n:
          KernelGemm.Both<{position == at : U32},KernelGemm.Both<{U32.is_eq(destination(context,position),query) == True{} : Bool},
            Misses(~Context,~destination,rest,(position + 1 : U32),context,query)>>
        case 1n+earlier:
          KernelGemm.Both<{U32.is_eq(destination(context,position),query) == False{} : Bool},Hits(~Context,~destination,rest,(position + 1 : U32),context,query,at,earlier)>

# Positions that never write query leave its value.
def missed(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+query: U32,+current: Element,
  misses: Misses(~Context,~destination,count,position,context,query)) ->
  {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,position,reader,context,query,current) == current : Element}:
  match count:
    case 0n: {==}
    case 1n+rest:
      KernelGemm.Both{+miss,rest_misses} = misses
      %Equal.sym(Bool,U32.is_eq(destination(context,position),query),False{},miss) :
        {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,
          Bool.pick(Element,_,combine(current,sample(reader,context,position)),current)) == current : Element}
      missed(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,current,rest_misses)

# One position writing query leaves the value combined with its sample.
def hit(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+query: U32,+at: U32,before: Nat,+current: Element,
  hits: Hits(~Context,~destination,count,position,context,query,at,before)) ->
  {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,position,reader,context,query,current) == combine(current,sample(reader,context,at)) : Element}:
  match count:
    case 0n:
      Empty.absurd({current == combine(current,sample(reader,context,at)) : Element},hits)
    case 1n+rest:
      match before:
        case 0n:
          KernelGemm.Both{+same,found_rest} = hits
          KernelGemm.Both{+found,rest_misses} = found_rest
          %Equal.sym(Bool,U32.is_eq(destination(context,position),query),True{},found) :
            {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,
              Bool.pick(Element,_,combine(current,sample(reader,context,position)),current)) == combine(current,sample(reader,context,at)) : Element}
          %same : {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,
            combine(current,sample(reader,context,position))) == combine(current,sample(reader,context,_)) : Element}
          missed(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,combine(current,sample(reader,context,position)),rest_misses)
        case 1n+earlier:
          KernelGemm.Both{+miss,rest_hits} = hits
          %Equal.sym(Bool,U32.is_eq(destination(context,position),query),False{},miss) :
            {updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,
              Bool.pick(Element,_,combine(current,sample(reader,context,position)),current)) == combine(current,sample(reader,context,at)) : Element}
          hit(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,query,at,earlier,current,rest_hits)

# ---- Runs ----

def kept_sample(~Element: Data,current: Element,sample: Element) -> Element:
  sample

# A run that stores its samples is the update that keeps the sample.
def run_as_update(~Model: Data,~Element: Data,~Context: Data,~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,count: Nat,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_step(~Model,~Element,~Context,~sample,~destination),count,state) ==
    Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~ArrayLaws.model_update_step(~Model,~Element,~Element,~Context,~sample,~kept_sample(~Element),~destination),count,state) :
    Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Traversal.State{+reader,+output,+position,+context} = state
      run_as_update(~Model,~Element,~Context,~sample,~destination,rest,
        Traversal.State{reader,Tree.write(Element,output,destination(context,position),sample(reader,context,position)),(position + 1 : U32),context})

# ---- The shape of the tree ----

def update_capacity(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
      ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,position,context}))) == Tree.capacity(Element,tree) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +address = destination(context,position)
      +value = combine(Tree.read(Element,tree,address),sample(reader,context,position))
      +written = Tree.write(Element,tree,address,value)
      Equal.trans(U32,
        Tree.capacity(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
          ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),rest,Traversal.State{reader,written,(position + 1 : U32),context}))),
        Tree.capacity(Element,written),Tree.capacity(Element,tree),
        update_capacity(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,written),
        StorageReadback.write_capacity(~Element,tree,address,value))

def update_balanced(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+tree: Tree.Tree<Element>,+depth: Nat,balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
    ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,position,context}))):
  match count:
    case 0n: balanced
    case 1n+rest:
      +address = destination(context,position)
      +value = combine(Tree.read(Element,tree,address),sample(reader,context,position))
      update_balanced(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,Tree.write(Element,tree,address,value),depth,
        Shape.write(Element,depth,tree,address,value,balanced))

# The loop keeps its reader.
def update_reader(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,+position: U32,+reader: Model,+context: Context,+tree: Tree.Tree<Element>) ->
  {ArrayLaws.model_reader(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
      ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,position,context})) == reader : Model}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +address = destination(context,position)
      update_reader(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,rest,(position + 1 : U32),reader,context,
        Tree.write(Element,tree,address,combine(Tree.read(Element,tree,address),sample(reader,context,position))))

# ---- Runs that write each position at itself ----

# Two numbers up to cap name different words when they differ.
def words_apart(+left: Nat,+right: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +left_room: {Nat.is_le(left,U32.to_nat(cap)) == True{} : Bool},+right_room: {Nat.is_le(right,U32.to_nat(cap)) == True{} : Bool},
  +apart: {Nat.is_eq(left,right) == False{} : Bool}) -> {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(left),U32.from_nat(right)),Nat.is_eq(U32.to_nat(U32.from_nat(left)),U32.to_nat(U32.from_nat(right))),
    Comparison.eq(U32.from_nat(left),U32.from_nat(right))) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,Bounds.roundtrip(left,cap,cap_ok,left_room)) : {Nat.is_eq(_,U32.to_nat(U32.from_nat(right))) == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(right)),right,Bounds.roundtrip(right,cap,cap_ok,right_room)) : {Nat.is_eq(left,_) == False{} : Bool}
  apart

# Positions from position on, past the query, never write it.
def identity_misses(~Context: Data,~destination: Context -> U32 -> U32,~identity: @+context: Context -> @+position: U32 -> {destination(context,position) == position : U32},
  count: Nat,+position: Nat,+context: Context,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +after: {Nat.is_lt(query,position) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(cap)) == True{} : Bool}) ->
  Misses(~Context,~destination,count,U32.from_nat(position),context,U32.from_nat(query)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +position_room = Bounds.strict_le(position,U32.to_nat(cap),Order.room_index(position,rest,U32.to_nat(cap),room))
      +query_room = Order.transitive(query,position,U32.to_nat(cap),Bounds.strict_le(query,position,after),position_room)
      KernelGemm.Both{
        %Equal.sym(U32,destination(context,U32.from_nat(position)),U32.from_nat(position),identity(context,U32.from_nat(position))) :
          {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
        words_apart(position,query,cap,cap_ok,position_room,query_room,
          %Equal.sym(Bool,Nat.is_eq(position,query),Nat.is_eq(query,position),Order.equal_symmetric(position,query)) : {_ == False{} : Bool}
          Order.less_not_equal(query,position,after)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Misses(~Context,~destination,rest,_,context,U32.from_nat(query))
        identity_misses(~Context,~destination,~identity,rest,1n+position,context,query,cap,cap_ok,Order.lt_succ(query,position,Bounds.strict_le(query,position,after)),
          Order.advance_room(position,rest,U32.to_nat(cap),room))}

# Positions from position on that write themselves miss a query past them.
def identity_past(~Context: Data,~destination: Context -> U32 -> U32,~identity: @+context: Context -> @+position: U32 -> {destination(context,position) == position : U32},
  count: Nat,+position: Nat,+context: Context,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +past: {Nat.is_le(Nat.add(position,count),query) == True{} : Bool},+query_room: {Nat.is_le(query,U32.to_nat(cap)) == True{} : Bool}) ->
  Misses(~Context,~destination,count,U32.from_nat(position),context,U32.from_nat(query)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +below = Order.lt_le(position,Nat.add(position,1n+rest),query,Order.below_sum(position,rest),past)
      +position_room = Bounds.strict_le(position,U32.to_nat(cap),Order.lt_le(position,query,U32.to_nat(cap),below,query_room))
      KernelGemm.Both{
        %Equal.sym(U32,destination(context,U32.from_nat(position)),U32.from_nat(position),identity(context,U32.from_nat(position))) :
          {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
        words_apart(position,query,cap,cap_ok,position_room,query_room,Order.less_not_equal(position,query,below)),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Misses(~Context,~destination,rest,_,context,U32.from_nat(query))
        identity_past(~Context,~destination,~identity,rest,1n+position,context,query,cap,cap_ok,
          %Addition.successor_right(position,rest) : {Nat.is_le(_,query) == True{} : Bool} past,
          query_room)}

# Of the positions from position on, only the query writes it.
def identity_hits(~Context: Data,~destination: Context -> U32 -> U32,~identity: @+context: Context -> @+position: U32 -> {destination(context,position) == position : U32},
  count: Nat,+position: Nat,before: Nat,+context: Context,+query: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +query_exact: {Nat.add(before,position) == query : Nat},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(before,count) == True{} : Bool}) ->
  Hits(~Context,~destination,count,U32.from_nat(position),context,U32.from_nat(query),U32.from_nat(query),before):
  match count:
    case 0n: Order.below_zero(before,inside)
    case 1n+ +rest:
      match before:
        case 0n:
          KernelGemm.Both{Equal.cong(Nat,U32,value => U32.from_nat(value),position,query,query_exact),
            KernelGemm.Both{
              %Equal.sym(U32,destination(context,U32.from_nat(position)),U32.from_nat(position),identity(context,U32.from_nat(position))) :
                {U32.is_eq(_,U32.from_nat(query)) == True{} : Bool}
              %query_exact : {U32.is_eq(U32.from_nat(position),U32.from_nat(_)) == True{} : Bool}
              Arithmetic.equal_reflexive(U32.from_nat(position)),
              %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
                Misses(~Context,~destination,rest,_,context,U32.from_nat(query))
              identity_misses(~Context,~destination,~identity,rest,1n+position,context,query,cap,cap_ok,
                %query_exact : {Nat.is_lt(_,1n+position) == True{} : Bool} Order.lt_succ(position,position,Order.reflexive(position)),
                Order.advance_room(position,rest,U32.to_nat(cap),room))}}
        case 1n+ +earlier:
          +position_room = Bounds.strict_le(position,U32.to_nat(cap),Order.room_index(position,rest,U32.to_nat(cap),room))
          +before_query = {%query_exact : {Nat.is_lt(position,_) == True{} : Bool}
            Order.lt_succ(position,Nat.add(earlier,position),%Order.commutative(position,earlier) : {Nat.is_le(position,_) == True{} : Bool} Order.le_add(position,earlier)) :
              {Nat.is_lt(position,query) == True{} : Bool}}
          +query_room = {%query_exact : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
            Order.transitive(Nat.add(1n+earlier,position),Nat.add(position,1n+rest),U32.to_nat(cap),
              %Order.commutative(position,1n+earlier) : {Nat.is_le(_,Nat.add(position,1n+rest)) == True{} : Bool}
              Order.offset_le(position,1n+earlier,1n+rest,Bounds.strict_le(1n+earlier,1n+rest,inside)),room) :
              {Nat.is_le(query,U32.to_nat(cap)) == True{} : Bool}}
          KernelGemm.Both{
            %Equal.sym(U32,destination(context,U32.from_nat(position)),U32.from_nat(position),identity(context,U32.from_nat(position))) :
              {U32.is_eq(_,U32.from_nat(query)) == False{} : Bool}
            words_apart(position,query,cap,cap_ok,position_room,query_room,Order.less_not_equal(position,query,before_query)),
            %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
              Hits(~Context,~destination,rest,_,context,U32.from_nat(query),U32.from_nat(query),earlier)
            identity_hits(~Context,~destination,~identity,rest,1n+position,earlier,context,query,cap,cap_ok,
              Equal.trans(Nat,Nat.add(earlier,1n+position),1n+Nat.add(earlier,position),query,Addition.successor_right(earlier,position),query_exact),
              Order.advance_room(position,rest,U32.to_nat(cap),room),inside)}

# The destinations of positions that write themselves lie below capacity.
def identity_destinations(~Context: Data,~destination: Context -> U32 -> U32,~identity: @+context: Context -> @+position: U32 -> {destination(context,position) == position : U32},
  count: Nat,+position: Nat,+context: Context,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  Destinations(~Context,~destination,count,U32.from_nat(position),context,capacity):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +below = Order.room_index(position,rest,U32.to_nat(capacity),room)
      KernelGemm.Both{
        %Equal.sym(U32,destination(context,U32.from_nat(position)),U32.from_nat(position),identity(context,U32.from_nat(position))) :
          {U32.is_lt(_,capacity) == True{} : Bool}
        Bounds.native_bound(position,capacity,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),below)),below),
        %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
          Destinations(~Context,~destination,rest,_,context,capacity)
        identity_destinations(~Context,~destination,~identity,rest,1n+position,context,capacity,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room))}

# A run of count positions that write themselves leaves at each of them its
# value combined with its sample.
def identity_read(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,~identity: @+context: Context -> @+position: U32 -> {destination(context,position) == position : U32},
  +count: Nat,+reader: Model,+context: Context,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +query: Nat,+inside: {Nat.is_lt(query,count) == True{} : Bool}) ->
  {Tree.read(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
      ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,0,context})),U32.from_nat(query)) ==
    combine(Tree.read(Element,tree,U32.from_nat(query)),sample(reader,context,U32.from_nat(query))) : Element}:
  +capacity = Tree.capacity(Element,tree)
  +query_below = Bounds.nat_trans(query,count,U32.to_nat(capacity),inside,room)
  Equal.trans(Element,
    Tree.read(Element,state_tree(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
      ~ArrayLaws.model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,Traversal.State{reader,tree,0,context})),U32.from_nat(query)),
    updated_at(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,0,reader,context,U32.from_nat(query),Tree.read(Element,tree,U32.from_nat(query))),
    combine(Tree.read(Element,tree,U32.from_nat(query)),sample(reader,context,U32.from_nat(query))),
    update_read(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,0,reader,context,tree,depth,U32.from_nat(query),balanced,limit,
      Bounds.native_bound(query,capacity,Bounds.roundtrip(query,capacity,capacity_ok,Bounds.strict_le(query,U32.to_nat(capacity),query_below)),query_below),
      identity_destinations(~Context,~destination,~identity,count,0n,context,capacity,capacity_ok,room)),
    hit(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,count,0,reader,context,U32.from_nat(query),U32.from_nat(query),query,Tree.read(Element,tree,U32.from_nat(query)),
      identity_hits(~Context,~destination,~identity,count,0n,query,context,query,capacity,capacity_ok,Addition.zero_right(query),room,inside)))
