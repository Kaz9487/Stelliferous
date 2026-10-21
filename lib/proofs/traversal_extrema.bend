# Filtered minimum/maximum folds bound every enabled sample they visit.
import Base
import ../traversal_array.bend as Traversal
import ./indexed_output_readback.bend as Readback
import ./u32_extrema.bend as Extrema
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./nat_extrema.bend as NaturalExtrema
import ./nat_to_u32_bounds.bend as Bounds
import ./traversal_array_refinement.bend as Arrays
import ./natural_addition.bend as Addition

type Query<-Context: Data> is Data:
  Query{context: Context,count: Nat,index: Nat}
def context(~Context: Data,query: Query<Context>) -> Context:
  Query{context,count,index} = query
  context
def count(~Context: Data,query: Query<Context>) -> Nat:
  Query{context,count,index} = query
  count
def index(~Context: Data,query: Query<Context>) -> Nat:
  Query{context,count,index} = query
  index

def bound(upper: Bool,+sample: U32,+value: U32) -> Data:
  Bool.pick(Data,upper,{U32.is_le(sample,value) == True{} : Bool},{U32.is_le(value,sample) == True{} : Bool})
def selected(upper: Bool,+value: U32,+sample: U32) -> U32:
  Bool.pick(U32,upper,U32.max(value,sample),U32.min(value,sample))

def wanted(~Context: Data,~sample: Context -> U32 -> U32,query: Query<Context>) -> U32:
  +query = query
  sample(context(~Context,query),U32.from_nat(index(~Context,query)))

def frame(~Context: Data,~used: Context -> U32 -> Bool,query: Query<Context>,value: U32) -> Data:
  +query = query
  {used(context(~Context,query),U32.from_nat(index(~Context,query))) == True{} : Bool}
def property(~upper: Bool,~Context: Data,~sample: Context -> U32 -> U32,query: Query<Context>,value: U32) -> Data:
  +query = query
  bound(upper,wanted(~Context,~sample,query),value)

def update(~upper: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,index: U32,query: Query<Context>,value: U32) -> U32:
  +query = query
  +index = index
  +context = context(~Context,query)
  +value = value
  Bool.pick(U32,used(context,index),selected(upper,value,sample(context,index)),value)

def frame_step(~upper: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,
  +query: Query<Context>,+index: U32,+value: U32,evidence: frame(~Context,~used,query,value)) -> frame(~Context,~used,query,update(~upper,~Context,~used,~sample,index,query,value)):
  evidence

def included(upper: Bool,+value: U32,+sample: U32) -> bound(upper,sample,selected(upper,value,sample)):
  match upper:
    case False{}: Extrema.minimum_right(value,sample)
    case True{}: Extrema.maximum_right(value,sample)

def at_target(~upper: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,
  +query: Query<Context>,+value: U32,evidence: frame(~Context,~used,query,value)) ->
  property(~upper,~Context,~sample,query,update(~upper,~Context,~used,~sample,U32.from_nat(index(~Context,query)),query,value)):
  %Equal.sym(Bool,used(context(~Context,query),U32.from_nat(index(~Context,query))),True{},evidence) :
    bound(upper,wanted(~Context,~sample,query),Bool.pick(U32,_,selected(upper,value,wanted(~Context,~sample,query)),value))
  included(upper,value,wanted(~Context,~sample,query))

def transitive(+left: U32,+middle: U32,+right: U32,lm: {U32.is_le(left,middle) == True{} : Bool},mr: {U32.is_le(middle,right) == True{} : Bool}) ->
  {U32.is_le(left,right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(left,right),Nat.is_le(U32.to_nat(left),U32.to_nat(right)),Comparison.le(left,right)) : {_ == True{} : Bool}
  Order.transitive(U32.to_nat(left),U32.to_nat(middle),U32.to_nat(right),Bounds.as_nat_le(left,middle,lm),Bounds.as_nat_le(middle,right,mr))

def kept_selected(upper: Bool,used: Bool,+wanted: U32,+value: U32,+sample: U32,evidence: bound(upper,wanted,value)) ->
  bound(upper,wanted,Bool.pick(U32,used,selected(upper,value,sample),value)):
  match upper used:
    case False{} False{}: evidence
    case True{} False{}: evidence
    case False{} True{}: transitive(U32.min(value,sample),value,wanted,Extrema.minimum_left(value,sample),evidence)
    case True{} True{}: transitive(wanted,value,U32.max(value,sample),evidence,Extrema.maximum_left(value,sample))

def post_step(~upper: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,
  +query: Query<Context>,+index: Nat,+value: U32,inside: {Nat.is_lt(index,count(~Context,query)) == True{} : Bool},evidence: property(~upper,~Context,~sample,query,value)) ->
  property(~upper,~Context,~sample,query,update(~upper,~Context,~used,~sample,U32.from_nat(index),query,value)):
  kept_selected(upper,used(context(~Context,query),U32.from_nat(index)),wanted(~Context,~sample,query),value,sample(context(~Context,query),U32.from_nat(index)),evidence)

def contains(~upper: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,
  +query: Query<Context>,+value: U32,evidence: frame(~Context,~used,query,value),inside: {Nat.is_lt(index(~Context,query),count(~Context,query)) == True{} : Bool}) ->
  property(~upper,~Context,~sample,query,Traversal.indexed_fold(~U32,~Query<Context>,~update(~upper,~Context,~used,~sample),count(~Context,query),0,query,value)):
  Readback.complete_postcondition(~U32,~Query<Context>,~count(~Context),~index(~Context),~frame(~Context,~used),~property(~upper,~Context,~sample),~update(~upper,~Context,~used,~sample),
    ~frame_step(~upper,~Context,~used,~sample),~at_target(~upper,~Context,~used,~sample),~post_step(~upper,~Context,~used,~sample),query,value,evidence,inside)

# A common natural envelope for all enabled samples is preserved from the
# initial value; it covers unsigned metadata representing padded origins.
# The caller discharges the sample bound for enabled indices, once per walk.
def natural_bound(higher: Bool,+value: U32,+boundary: Nat) -> Data:
  Bool.pick(Data,higher,{Nat.is_le(U32.to_nat(value),boundary) == True{} : Bool},{Nat.is_le(boundary,U32.to_nat(value)) == True{} : Bool})
def natural_frame(~higher: Bool,~Context: Data,~boundary: Context -> Nat,context: Context,index: Nat,value: U32) -> Data:
  +context = context
  +value = value
  natural_bound(higher,value,boundary(context))
def filtered(~higher: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,index: U32,context: Context,value: U32) -> U32:
  +context = context
  +index = index
  +value = value
  Bool.pick(U32,used(context,index),selected(higher,value,sample(context,index)),value)

def natural_selected(higher: Bool,+value: U32,+sample: U32,+boundary: Nat,
  before: natural_bound(higher,value,boundary),sample_bound: natural_bound(higher,sample,boundary)) ->
  natural_bound(higher,selected(higher,value,sample),boundary):
  match higher:
    case False{}:
      %Equal.sym(Nat,U32.to_nat(U32.min(value,sample)),Nat.min(U32.to_nat(value),U32.to_nat(sample)),Extrema.minimum(value,sample)) : {Nat.is_le(boundary,_) == True{} : Bool}
      NaturalExtrema.minimum_lower(U32.to_nat(value),U32.to_nat(sample),boundary,before,sample_bound)
    case True{}:
      %Equal.sym(Nat,U32.to_nat(U32.max(value,sample)),Nat.max(U32.to_nat(value),U32.to_nat(sample)),Extrema.maximum(value,sample)) : {Nat.is_le(_,boundary) == True{} : Bool}
      NaturalExtrema.maximum_upper(U32.to_nat(value),U32.to_nat(sample),boundary,before,sample_bound)

def natural_enabled(~higher: Bool,~Context: Data,~count: Context -> Nat,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,~boundary: Context -> Nat,
  ~sample_bound: @+context: Context -> @+index: Nat -> {Nat.is_lt(index,count(context)) == True{} : Bool} ->
    {used(context,U32.from_nat(index)) == True{} : Bool} -> natural_bound(higher,sample(context,U32.from_nat(index)),boundary(context)),
  enabled: Bool,+context: Context,+index: Nat,+value: U32,+inside: {Nat.is_lt(index,count(context)) == True{} : Bool},
  compared: {used(context,U32.from_nat(index)) == enabled : Bool},before: natural_frame(~higher,~Context,~boundary,context,index,value)) ->
  natural_frame(~higher,~Context,~boundary,context,1n+index,Bool.pick(U32,enabled,selected(higher,value,sample(context,U32.from_nat(index))),value)):
  match enabled:
    case False{}: before
    case True{}: natural_selected(higher,value,sample(context,U32.from_nat(index)),boundary(context),before,sample_bound(context,index,inside,compared))

def natural_step(~higher: Bool,~Context: Data,~count: Context -> Nat,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,~boundary: Context -> Nat,
  ~sample_bound: @+context: Context -> @+index: Nat -> {Nat.is_lt(index,count(context)) == True{} : Bool} ->
    {used(context,U32.from_nat(index)) == True{} : Bool} -> natural_bound(higher,sample(context,U32.from_nat(index)),boundary(context)),
  +context: Context,+index: Nat,+value: U32,inside: {Nat.is_lt(index,count(context)) == True{} : Bool},before: natural_frame(~higher,~Context,~boundary,context,index,value)) ->
  natural_frame(~higher,~Context,~boundary,context,1n+index,filtered(~higher,~Context,~used,~sample,U32.from_nat(index),context,value)):
  +inside = inside
  natural_enabled(~higher,~Context,~count,~used,~sample,~boundary,~sample_bound,used(context,U32.from_nat(index)),context,index,value,inside,{==},before)

def natural_fold(~higher: Bool,~Context: Data,~count: Context -> Nat,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,~boundary: Context -> Nat,
  ~sample_bound: @+context: Context -> @+index: Nat -> {Nat.is_lt(index,count(context)) == True{} : Bool} ->
    {used(context,U32.from_nat(index)) == True{} : Bool} -> natural_bound(higher,sample(context,U32.from_nat(index)),boundary(context)),
  +context: Context,+initial: U32,evidence: natural_frame(~higher,~Context,~boundary,context,0n,initial)) ->
  natural_frame(~higher,~Context,~boundary,context,count(context),Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),count(context),0,context,initial)):
  Arrays.indexed_fold_preserves_invariant(~U32,~Context,~count,~natural_frame(~higher,~Context,~boundary),~filtered(~higher,~Context,~used,~sample),~natural_step(~higher,~Context,~count,~used,~sample,~boundary,~sample_bound),count(context),0n,context,initial,Order.reflexive(count(context)),evidence)

# A filtered fold keeps its initial value unless it visits an enabled index.
type Enabled<-Context: Data,-used: Context -> U32 -> Bool,-context: Context,-count: Nat> is Data:
  Enabled{index: Nat,inside: {Nat.is_lt(index,count) == True{} : Bool},enabled: {used(context,U32.from_nat(index)) == True{} : Bool}}

def initial_or_enabled_from(~higher: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,count: Nat,enabled: Bool,+first: Nat,+context: Context,+value: U32,
  +compared: {used(context,U32.from_nat(first)) == enabled : Bool}) ->
  Either<&2,&2,{Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),count,U32.from_nat(first),context,value) == value : U32},
    Enabled<Context,used,context,Nat.add(first,count)>>:
  match count enabled:
    case 0n True{}: Inl{{==}}
    case 0n False{}: Inl{{==}}
    case 1n+ +rest True{}:
      Inr{Enabled{first,Order.room_index(first,rest,Nat.add(first,1n+rest),Order.reflexive(Nat.add(first,1n+rest))),compared}}
    case 1n+ +rest False{}:
      %Equal.sym(Bool,used(context,U32.from_nat(first)),False{},compared) :
        Either<&2,&2,{Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),rest,(U32.from_nat(first) + 1 : U32),context,
          Bool.pick(U32,_,selected(higher,value,sample(context,U32.from_nat(first))),value)) == value : U32},Enabled<Context,used,context,Nat.add(first,1n+rest)>>
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        Either<&2,&2,{Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),rest,_,context,value) == value : U32},Enabled<Context,used,context,Nat.add(first,1n+rest)>>
      %Equal.sym(Nat,Nat.add(first,1n+rest),1n+Nat.add(first,rest),Addition.successor_right(first,rest)) :
        Either<&2,&2,{Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),rest,U32.from_nat(1n+first),context,value) == value : U32},Enabled<Context,used,context,_>>
      initial_or_enabled_from(~higher,~Context,~used,~sample,rest,used(context,U32.from_nat(1n+first)),1n+first,context,value,{==})

def initial_or_enabled(~higher: Bool,~Context: Data,~used: Context -> U32 -> Bool,~sample: Context -> U32 -> U32,count: Nat,+first: Nat,+context: Context,+value: U32) ->
  Either<&2,&2,{Traversal.indexed_fold(~U32,~Context,~filtered(~higher,~Context,~used,~sample),count,U32.from_nat(first),context,value) == value : U32},
    Enabled<Context,used,context,Nat.add(first,count)>>:
  initial_or_enabled_from(~higher,~Context,~used,~sample,count,used(context,U32.from_nat(first)),first,context,value,{==})
