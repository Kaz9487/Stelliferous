# A left fold of samples, preserving their exact order and the reader owner.
# Both whole and band window readers instantiate this common proof.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_array_refinement.bend as ArrayLaws
import ../kernel_gemm.bend as KernelGemm

def values(~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  count: Nat,+position: U32,+source: Source,+context: Context,accumulator: Accumulator) -> Accumulator:
  match count:
    case 0n: accumulator
    case 1n+rest:
      values(~Source,~Element,~Accumulator,~Context,~sample,~operation,rest,(position + 1 : U32),source,context,operation(accumulator,sample(source,context,position)))

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

# The fold one value a step is the ordered fold.
def fold(~Reader: Type,~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack: Source -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  ~read_matches: @+source: Source -> @+context: Context -> @+position: U32 -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & Element},
  +count: Nat,+position: U32,+source: Source,+context: Context,+initial: Accumulator) ->
  {Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context) ==
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)) : Reader & Accumulator}:
  +state = {Traversal.State{source,initial,position,context} : Traversal.State<&2,&2,Source,Accumulator,Context>}
  Equal.trans(Reader & Accumulator,
    Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context),
    Traversal.finish_fold(~Reader,~Accumulator,~Context,ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,
      Loop.run(~Traversal.State<&2,&2,Source,Accumulator,Context>,~ArrayLaws.model_fold_step(~Source,~Element,~Accumulator,~Context,~sample,~operation),count,state))),
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)),
    Equal.cong(Traversal.State<&1,&2,Reader,Accumulator,Context>,Reader & Accumulator,folded => Traversal.finish_fold(~Reader,~Accumulator,~Context,folded),
      Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~operation),count,
        ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,state)),
      ArrayLaws.represent_fold(~Reader,~Source,~Accumulator,~Context,~pack,
        Loop.run(~Traversal.State<&2,&2,Source,Accumulator,Context>,~ArrayLaws.model_fold_step(~Source,~Element,~Accumulator,~Context,~sample,~operation),count,state)),
      ArrayLaws.fold_refines_ordered_model(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~read,~sample,~operation,~read_matches,count,state)),
    modeled(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~sample,~operation,count,position,source,context,initial))
