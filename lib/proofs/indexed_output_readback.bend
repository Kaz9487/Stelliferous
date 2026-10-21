# One coverage argument for any indexed output traversal. Each query names its
# producing job; that job writes the specified value and later jobs preserve it.
# A concrete strategy must discharge these local obligations and target bounds.
import Base
import ../traversal_array.bend as Traversal
import ./traversal_array_refinement.bend as ArrayLaws
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds

def suffix_preserves_readback(~State: Data,~Context: Data,~Value: Data,
  ~limit: Context -> Nat,~target: Context -> Nat,~Frame: Context -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) ->
    {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +count: Nat,+first: Nat,+context: Context,+state: State,+frame: Frame(context,state),
  +after: {Nat.is_lt(target(context),first) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(first,count),limit(context)) == True{} : Bool}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state)) == read(context,state) : Value}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,_,context,update(U32.from_nat(first),context,state))) == read(context,state) : Value}
      %later(context,first,state,frame,after,Order.room_index(first,rest,limit(context),room)) :
        {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+first),context,update(U32.from_nat(first),context,state))) == _ : Value}
      suffix_preserves_readback(~State,~Context,~Value,~limit,~target,~Frame,~read,~update,~frame_step,~later,
        rest,1n+first,context,update(U32.from_nat(first),context,state),frame_step(context,U32.from_nat(first),state,frame),
        Order.lt_succ(target(context),first,Bounds.strict_le(target(context),first,after)),Order.advance_room(first,rest,limit(context),room))

def prefix_preserves_frame(~State: Data,~Context: Data,
  ~limit: Context -> Nat,~Frame: Context -> State -> Data,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  +count: Nat,+context: Context,+state: State,room: {Nat.is_le(count,limit(context)) == True{} : Bool},frame: Frame(context,state)) ->
  Frame(context,Traversal.indexed_fold(~State,~Context,~update,count,0,context,state)):
  ArrayLaws.indexed_fold_preserves_invariant(~State,~Context,~limit,~(context => index => state => Frame(context,state)),~update,
    ~(context => index => state => inside => evidence => frame_step(context,U32.from_nat(index),state,evidence)),count,0n,context,state,room,frame)

def suffix_room(+total: Nat,+index: Nat,inside: {Nat.is_lt(index,total) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+index,Nat.sub(total,1n+index)),total) == True{} : Bool}:
  %Addition.successor_right(index,Nat.sub(total,1n+index)) :
    {Nat.is_le(_,total) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(index,1n+Nat.sub(total,1n+index)),total,Order.split_at_element(total,index,inside)) :
    {Nat.is_le(_,total) == True{} : Bool}
  Order.reflexive(total)

# No FP32-specific reasoning, lane enumeration, or second whole-loop proof is
# required when a new execution strategy uses the common indexed traversal.
def complete_readback(~State: Data,~Context: Data,~Value: Data,
  ~limit: Context -> Nat,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~at_target: @+context: Context -> @+state: State -> Frame(context,state) ->
    {read(context,update(U32.from_nat(target(context)),context,state)) == wanted(context) : Value},
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) ->
    {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +context: Context,+state: State,frame: Frame(context,state),+inside: {Nat.is_lt(target(context),limit(context)) == True{} : Bool}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,limit(context),0,context,state)) == wanted(context) : Value}:
  +index = target(context)
  +rest = Nat.sub(limit(context),1n+index)
  +prefix = Traversal.indexed_fold(~State,~Context,~update,index,0,context,state)
  +ready = prefix_preserves_frame(~State,~Context,~limit,~Frame,~update,~frame_step,index,context,state,Bounds.strict_le(index,limit(context),inside),frame)
  +written = update(U32.from_nat(index),context,prefix)
  %Equal.sym(Nat,limit(context),Nat.add(index,1n+rest),Equal.sym(Nat,Nat.add(index,1n+rest),limit(context),Order.split_at_element(limit(context),index,inside))) :
    {read(context,Traversal.indexed_fold(~State,~Context,~update,_,0,context,state)) == wanted(context) : Value}
  %Equal.sym(State,Traversal.indexed_fold(~State,~Context,~update,Nat.add(index,1n+rest),0,context,state),
    Traversal.indexed_fold(~State,~Context,~update,1n+rest,U32.from_nat(index),context,prefix),
    ArrayLaws.indexed_fold_split(~State,~Context,~update,index,1n+rest,0n,context,state)) :
    {read(context,_) == wanted(context) : Value}
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
    {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,_,context,written)) == wanted(context) : Value}
  Equal.trans(Value,read(context,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+index),context,written)),read(context,written),wanted(context),
    suffix_preserves_readback(~State,~Context,~Value,~limit,~target,~Frame,~read,~update,~frame_step,~later,rest,1n+index,context,written,
      frame_step(context,U32.from_nat(index),prefix,ready),Order.lt_succ(index,index,Order.reflexive(index)),suffix_room(limit(context),index,inside)),
    at_target(context,prefix,ready))

# A producer can establish a postcondition that subsequent jobs preserve even
# when they may write the same value again. All traversal reasoning is supplied
# by the existing split and invariant lemmas.
def complete_postcondition(~State: Data,~Context: Data,
  ~limit: Context -> Nat,~target: Context -> Nat,
  ~Frame: Context -> State -> Data,~Post: Context -> State -> Data,
  ~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~at_target: @+context: Context -> @+state: State -> Frame(context,state) -> Post(context,update(U32.from_nat(target(context)),context,state)),
  ~post_step: @+context: Context -> @+index: Nat -> @+state: State ->
    {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Post(context,state) -> Post(context,update(U32.from_nat(index),context,state)),
  +context: Context,+state: State,frame: Frame(context,state),+inside: {Nat.is_lt(target(context),limit(context)) == True{} : Bool}) ->
  Post(context,Traversal.indexed_fold(~State,~Context,~update,limit(context),0,context,state)):
  +index = target(context)
  +rest = Nat.sub(limit(context),1n+index)
  +prefix = Traversal.indexed_fold(~State,~Context,~update,index,0,context,state)
  +ready = prefix_preserves_frame(~State,~Context,~limit,~Frame,~update,~frame_step,index,context,state,Bounds.strict_le(index,limit(context),inside),frame)
  +written = update(U32.from_nat(index),context,prefix)
  %Equal.sym(Nat,limit(context),Nat.add(index,1n+rest),Equal.sym(Nat,Nat.add(index,1n+rest),limit(context),Order.split_at_element(limit(context),index,inside))) :
    Post(context,Traversal.indexed_fold(~State,~Context,~update,_,0,context,state))
  %Equal.sym(State,Traversal.indexed_fold(~State,~Context,~update,Nat.add(index,1n+rest),0,context,state),
    Traversal.indexed_fold(~State,~Context,~update,1n+rest,U32.from_nat(index),context,prefix),
    ArrayLaws.indexed_fold_split(~State,~Context,~update,index,1n+rest,0n,context,state)) : Post(context,_)
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
    Post(context,Traversal.indexed_fold(~State,~Context,~update,rest,_,context,written))
  ArrayLaws.indexed_fold_preserves_invariant(~State,~Context,~limit,~(context => index => state => Post(context,state)),~update,~post_step,
    rest,1n+index,context,written,suffix_room(limit(context),index,inside),at_target(context,prefix,ready))

# A disjoint bounded traversal transports an existing observation from its start.
type Kept<-State: Data,-Context: Data,-Value: Data,-read: Context -> State -> Value,-wanted: Context -> Value,-Frame: Context -> State -> Data,-context: Context,-state: State> is Data:
  Kept{frame: Frame(context,state),value: {read(context,state) == wanted(context) : Value}}

def kept_step(~State: Data,~Context: Data,~Value: Data,~limit: Context -> Nat,~read: Context -> State -> Value,~wanted: Context -> Value,~Frame: Context -> State -> Data,
  ~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~unchanged: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +context: Context,+index: Nat,+state: State,inside: {Nat.is_lt(index,limit(context)) == True{} : Bool},
  evidence: Kept<State,Context,Value,read,wanted,Frame,context,state>) ->
  Kept<State,Context,Value,read,wanted,Frame,context,update(U32.from_nat(index),context,state)>:
  Kept{+frame,value} = evidence
  Kept{frame_step(context,U32.from_nat(index),state,frame),
    Equal.trans(Value,read(context,update(U32.from_nat(index),context,state)),read(context,state),wanted(context),unchanged(context,index,state,frame,inside),value)}

def kept_value(~State: Data,~Context: Data,~Value: Data,~read: Context -> State -> Value,~wanted: Context -> Value,~Frame: Context -> State -> Data,
  context: Context,state: State,evidence: Kept<State,Context,Value,read,wanted,Frame,context,state>) -> {read(context,state) == wanted(context) : Value}:
  Kept{frame,value} = evidence
  value

def preserves_readback(~State: Data,~Context: Data,~Value: Data,~limit: Context -> Nat,~read: Context -> State -> Value,~wanted: Context -> Value,~Frame: Context -> State -> Data,
  ~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~unchanged: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +count: Nat,+first: Nat,+context: Context,+state: State,
  room: {Nat.is_le(Nat.add(first,count),limit(context)) == True{} : Bool},frame: Frame(context,state),initial: {read(context,state) == wanted(context) : Value}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state)) == wanted(context) : Value}:
  kept_value(~State,~Context,~Value,~read,~wanted,~Frame,context,Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state),
    ArrayLaws.indexed_fold_preserves_invariant(~State,~Context,~limit,~(context => index => state => Kept<State,Context,Value,read,wanted,Frame,context,state>),~update,
      ~kept_step(~State,~Context,~Value,~limit,~read,~wanted,~Frame,~update,~frame_step,~unchanged),count,first,context,state,room,Kept{frame,initial}))

# A position-dependent frame can track packed prefix bases while carrying a
# readback after its producer. The existing indexed invariant supplies the loop.
def completed(~State: Data,~Context: Data,~Value: Data,~target: Context -> Nat,~wanted: Context -> Value,~read: Context -> State -> Value,
  context: Context,index: Nat,state: State) -> Data:
  +context = context
  Bool.pick(Data,Nat.is_lt(target(context),index),{read(context,state) == wanted(context) : Value},Unit)

type IndexedKept<-State: Data,-Context: Data,-Value: Data,-target: Context -> Nat,-wanted: Context -> Value,-Frame: Context -> Nat -> State -> Data,-read: Context -> State -> Value,-context: Context,-index: Nat,-state: State> is Data:
  IndexedKept{frame: Frame(context,index,state),done: completed(State,Context,Value,target,wanted,read,context,index,state)}

def indexed_cases(~State: Data,~Context: Data,~Value: Data,~limit: Context -> Nat,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> Nat -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: Nat -> @+state: State -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Frame(context,index,state) -> Frame(context,1n+index,update(U32.from_nat(index),context,state)),
  ~produced: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {index == target(context) : Nat} -> {read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  comparison: Cmp,+context: Context,+index: Nat,+state: State,+inside: {Nat.is_lt(index,limit(context)) == True{} : Bool},
  evidence: IndexedKept<State,Context,Value,target,wanted,Frame,read,context,index,state>,
  compared: {Nat.cmp(target(context),index) == comparison : Cmp}) ->
  IndexedKept<State,Context,Value,target,wanted,Frame,read,context,1n+index,update(U32.from_nat(index),context,state)>:
  match comparison:
    case LT{}:
      IndexedKept{+frame,done} = evidence
      +after = { %Equal.sym(Cmp,Nat.cmp(target(context),index),LT{},compared) : {Cmp.is_lt(_) == True{} : Bool} {==} : {Nat.is_lt(target(context),index) == True{} : Bool} }
      +next_after = Order.lt_succ(target(context),index,Bounds.strict_le(target(context),index,after))
      IndexedKept{frame_step(context,index,state,inside,frame),
        %Equal.sym(Bool,Nat.is_lt(target(context),1n+index),True{},next_after) : Bool.pick(Data,_,{read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},Unit)
        Equal.trans(Value,read(context,update(U32.from_nat(index),context,state)),read(context,state),wanted(context),later(context,index,state,frame,after,inside),
          %after : Bool.pick(Data,_,{read(context,state) == wanted(context) : Value},Unit) done)}
    case EQ{}:
      IndexedKept{+frame,done} = evidence
      +same = Order.equal_of_is_eq(target(context),index,
        %Equal.sym(Cmp,Nat.cmp(target(context),index),EQ{},compared) : {Cmp.is_eq(_) == True{} : Bool} {==})
      +after = { %Equal.sym(Nat,target(context),index,same) : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) : {Nat.is_lt(target(context),1n+index) == True{} : Bool} }
      IndexedKept{frame_step(context,index,state,inside,frame),
        %Equal.sym(Bool,Nat.is_lt(target(context),1n+index),True{},after) : Bool.pick(Data,_,{read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},Unit)
        produced(context,index,state,frame,Equal.sym(Nat,target(context),index,same))}
    case GT{}:
      IndexedKept{frame,done} = evidence
      +before = Order.ge_false(index,target(context),
        %Equal.sym(Bool,Nat.is_ge(index,target(context)),Nat.is_le(target(context),index),Order.ge_le(index,target(context))) : {_ == False{} : Bool}
          %Equal.sym(Cmp,Nat.cmp(target(context),index),GT{},compared) : {Cmp.is_le(_) == False{} : Bool} {==})
      +not_after = Order.not_below(1n+index,target(context),Order.strict_follow(index,target(context),before))
      IndexedKept{frame_step(context,index,state,inside,frame),
        %Equal.sym(Bool,Nat.is_lt(target(context),1n+index),False{},not_after) : Bool.pick(Data,_,{read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},Unit)
        Unit{}}

def indexed_step(~State: Data,~Context: Data,~Value: Data,~limit: Context -> Nat,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> Nat -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: Nat -> @+state: State -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Frame(context,index,state) -> Frame(context,1n+index,update(U32.from_nat(index),context,state)),
  ~produced: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {index == target(context) : Nat} -> {read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +context: Context,+index: Nat,+state: State,inside: {Nat.is_lt(index,limit(context)) == True{} : Bool},
  evidence: IndexedKept<State,Context,Value,target,wanted,Frame,read,context,index,state>) ->
  IndexedKept<State,Context,Value,target,wanted,Frame,read,context,1n+index,update(U32.from_nat(index),context,state)>:
  indexed_cases(~State,~Context,~Value,~limit,~target,~wanted,~Frame,~read,~update,~frame_step,~produced,~later,Nat.cmp(target(context),index),context,index,state,inside,evidence,{==})

def indexed_value(~State: Data,~Context: Data,~Value: Data,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> Nat -> State -> Data,~read: Context -> State -> Value,
  +context: Context,+index: Nat,+state: State,evidence: IndexedKept<State,Context,Value,target,wanted,Frame,read,context,index,state>,after: {Nat.is_lt(target(context),index) == True{} : Bool}) ->
  {read(context,state) == wanted(context) : Value}:
  IndexedKept{frame,done} = evidence
  %after : Bool.pick(Data,_,{read(context,state) == wanted(context) : Value},Unit)
  done

def indexed_complete(~State: Data,~Context: Data,~Value: Data,~limit: Context -> Nat,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> Nat -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: Nat -> @+state: State -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Frame(context,index,state) -> Frame(context,1n+index,update(U32.from_nat(index),context,state)),
  ~produced: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {index == target(context) : Nat} -> {read(context,update(U32.from_nat(index),context,state)) == wanted(context) : Value},
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,index,state) -> {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} -> {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +context: Context,+state: State,frame: Frame(context,0n,state),inside: {Nat.is_lt(target(context),limit(context)) == True{} : Bool}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,limit(context),0,context,state)) == wanted(context) : Value}:
  +initial = {IndexedKept{frame,
    %Equal.sym(Bool,Nat.is_lt(target(context),0n),False{},Order.not_below(0n,target(context),Order.zero_le(target(context)))) : Bool.pick(Data,_,{read(context,state) == wanted(context) : Value},Unit)
      Unit{}} : IndexedKept<State,Context,Value,target,wanted,Frame,read,context,0n,state>}
  indexed_value(~State,~Context,~Value,~target,~wanted,~Frame,~read,context,limit(context),Traversal.indexed_fold(~State,~Context,~update,limit(context),0,context,state),
    ArrayLaws.indexed_fold_preserves_invariant(~State,~Context,~limit,~(context => index => state => IndexedKept<State,Context,Value,target,wanted,Frame,read,context,index,state>),~update,
      ~indexed_step(~State,~Context,~Value,~limit,~target,~wanted,~Frame,~read,~update,~frame_step,~produced,~later),limit(context),0n,context,state,Order.reflexive(limit(context)),initial),inside)
