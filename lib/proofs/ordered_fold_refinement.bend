# A left fold of samples, preserving their exact order and the reader owner.
# Both whole and band window readers instantiate this common proof.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_shape.bend as Shape
import ./traversal_array_refinement.bend as ArrayLaws

def values(~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  count: Nat,+position: U32,+source: Source,+context: Context,accumulator: Accumulator) -> Accumulator:
  match count:
    case 0n: accumulator
    case 1n+rest:
      values(~Source,~Element,~Accumulator,~Context,~sample,~operation,rest,(position + 1 : U32),source,context,operation(accumulator,sample(source,context,position)))

def fold(~Reader: Type,~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack: Source -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  ~read_matches: @+source: Source -> @+context: Context -> @+position: U32 -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & Element},
  count: Nat,+position: U32,+source: Source,+context: Context,+initial: Accumulator) ->
  {Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context) ==
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)) : Reader & Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Reader & Element,read(pack(source),context,position),(pack(source),sample(source,context,position)),read_matches(source,context,position)) :
        {Traversal.finish_fold(~Reader,~Accumulator,~Context,
          Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~operation),rest,
            Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~operation,initial,position,context,_))) ==
          (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,1n+rest,position,source,context,initial)) : Reader & Accumulator}
      fold(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~read,~sample,~operation,~read_matches,rest,(position + 1 : U32),source,context,
        operation(initial,sample(source,context,position)))

# The model loop of the ordered fold keeps its source and folds values.
def modeled(~Reader: Type,~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,~pack: Source -> Reader,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  count: Nat,+position: U32,+source: Source,+context: Context,+accumulator: Accumulator) ->
  {Traversal.finish_fold(~Reader,~Accumulator,~Context,ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,
      Loop.run(~Traversal.State<&2,&2,Source,Accumulator,Context>,~ArrayLaws.model_fold_step(~Source,~Element,~Accumulator,~Context,~sample,~operation),count,
        Traversal.State{source,accumulator,position,context}))) ==
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,accumulator)) : Reader & Accumulator}:
  match count:
    case 0n: {==}
    case 1n+rest:
      modeled(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~sample,~operation,rest,(position + 1 : U32),source,context,
        operation(accumulator,sample(source,context,position)))

# The fold in rounds of eight is the same ordered fold.
def rounds(~Reader: Type,~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack: Source -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  ~read_matches: @+source: Source -> @+context: Context -> @+position: U32 -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & Element},
  +count: Nat,+position: U32,+source: Source,+context: Context,+initial: Accumulator) ->
  {Traversal.fold_rounds(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context) ==
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)) : Reader & Accumulator}:
  +state = {Traversal.State{source,initial,position,context} : Traversal.State<&2,&2,Source,Accumulator,Context>}
  Equal.trans(Reader & Accumulator,
    Traversal.fold_rounds(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context),
    Traversal.finish_fold(~Reader,~Accumulator,~Context,ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,
      Loop.run(~Traversal.State<&2,&2,Source,Accumulator,Context>,~ArrayLaws.model_fold_step(~Source,~Element,~Accumulator,~Context,~sample,~operation),count,state))),
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)),
    Equal.cong(Traversal.State<&1,&2,Reader,Accumulator,Context>,Reader & Accumulator,folded => Traversal.finish_fold(~Reader,~Accumulator,~Context,folded),
      Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~operation),Nat.div(count,8n),
        Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~operation),Nat.mod(count,8n),
          ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,state))),
      ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,
        Loop.run(~Traversal.State<&2,&2,Source,Accumulator,Context>,~ArrayLaws.model_fold_step(~Source,~Element,~Accumulator,~Context,~sample,~operation),count,state)),
      ArrayLaws.rounds_refine_ordered_model(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~read,~sample,~operation,~read_matches,count,state)),
    modeled(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~sample,~operation,count,position,source,context,initial))

# Two folds whose samples agree position by position.
def Agree(~Element: Data,~Left: Data,~LeftContext: Data,~left_sample: Left -> LeftContext -> U32 -> Element,
  ~Right: Data,~RightContext: Data,~right_sample: Right -> RightContext -> U32 -> Element,
  +left: Left,+left_context: LeftContext,+right: Right,+right_context: RightContext,remaining: Nat,+position: U32) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest:
      Shape.Both<{left_sample(left,left_context,position) == right_sample(right,right_context,position) : Element},
        Agree(~Element,~Left,~LeftContext,~left_sample,~Right,~RightContext,~right_sample,left,left_context,right,right_context,rest,(position + 1 : U32))>

# Folds whose samples agree fold to the same value.
def agreed(~Element: Data,~Accumulator: Data,~Left: Data,~LeftContext: Data,~left_sample: Left -> LeftContext -> U32 -> Element,
  ~Right: Data,~RightContext: Data,~right_sample: Right -> RightContext -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  remaining: Nat,+position: U32,+left: Left,+left_context: LeftContext,+right: Right,+right_context: RightContext,+initial: Accumulator,
  facts: Agree(~Element,~Left,~LeftContext,~left_sample,~Right,~RightContext,~right_sample,left,left_context,right,right_context,remaining,position)) ->
  {values(~Left,~Element,~Accumulator,~LeftContext,~left_sample,~operation,remaining,position,left,left_context,initial) ==
    values(~Right,~Element,~Accumulator,~RightContext,~right_sample,~operation,remaining,position,right,right_context,initial) : Accumulator}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Shape.Both{head,tail} = facts
      %Equal.sym(Element,left_sample(left,left_context,position),right_sample(right,right_context,position),head) :
        {values(~Left,~Element,~Accumulator,~LeftContext,~left_sample,~operation,rest,(position + 1 : U32),left,left_context,operation(initial,_)) ==
          values(~Right,~Element,~Accumulator,~RightContext,~right_sample,~operation,1n+rest,position,right,right_context,initial) : Accumulator}
      agreed(~Element,~Accumulator,~Left,~LeftContext,~left_sample,~Right,~RightContext,~right_sample,~operation,rest,(position + 1 : U32),left,left_context,right,right_context,
        operation(initial,right_sample(right,right_context,position)),tail)
