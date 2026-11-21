# The same ownership-parameterized state is used by implementation and model.
# One read contract and one generic Array.set refinement cover every traversal.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./loop_projection.bend as Projection
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./list_storage_readback.bend as ListRead
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../kernel_gemm.bend as KernelGemm

def collect_values_context(~ownership: Quant,~Runtime: Kind(ownership),~Model: Data,~Context: Data,~pack: Context -> Model -> Runtime,values: List<&2,Model>,+context: Context) -> List<ownership,Runtime>:
  match values:
    case Nil{}: []
    case head <> tail: pack(context,head) <> collect_values_context(~ownership,~Runtime,~Model,~Context,~pack,tail,context)

def collect_values(~ownership: Quant,~Runtime: Kind(ownership),~Model: Data,~pack: Model -> Runtime,values: List<&2,Model>) -> List<ownership,Runtime>:
  collect_values_context(~ownership,~Runtime,~Model,~Unit,~{context => pack : Unit -> Model -> Runtime},values,Unit{})

def collect_result(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,result: Source & List<&2,Model>) -> Reader & List<ownership,Runtime>:
  (+source,+values) = result
  (pack_reader(source),collect_values(~ownership,~Runtime,~Model,~pack_value,values))

def collect_prepend_source(~Source: Data,~Element: Data,+value: Element,result: Source & List<&2,Element>) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect_prepend(~&2,~Source,~Element,value,result)) == Pair.fst(Source,List<&2,Element>,result) : Source}:
  (source,values) = result
  {==}

def collect_list_next_source(~Source: Data,~Element: Data,head: Source & Element,tail: Source -> Source & List<&2,Element>,
  kept: @+source: Source -> {Pair.fst(Source,List<&2,Element>,tail(source)) == source : Source}) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect_list_next(~&2,~Source,~Element,head,tail)) == Pair.fst(Source,Element,head) : Source}:
  (+source,+value) = head
  Equal.trans(Source,Pair.fst(Source,List<&2,Element>,Traversal.collect_prepend(~&2,~Source,~Element,value,tail(source))),Pair.fst(Source,List<&2,Element>,tail(source)),source,
    collect_prepend_source(~Source,~Element,value,tail(source)),kept(source))

# A keyed collection whose reads keep the source keeps it.
def collect_list_preserves_source(~Source: Data,~Element: Data,~Key: Data,~Context: Data,
  ~read: Source -> Context -> Key -> Source & Element,
  ~kept: @+source: Source -> @+context: Context -> @+key: Key -> {Pair.fst(Source,Element,read(source,context,key)) == source : Source},
  keys: List<&2,Key>,+source: Source,+context: Context) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect_list(~&2,~Source,~Element,~Key,~Context,~read,keys,source,context)) == source : Source}:
  match keys:
    case Nil{}: {==}
    case Con{+key,+rest}:
      Equal.trans(Source,Pair.fst(Source,List<&2,Element>,Traversal.collect_list(~&2,~Source,~Element,~Key,~Context,~read,Con{key,rest},source,context)),
        Pair.fst(Source,Element,read(source,context,key)),source,
        collect_list_next_source(~Source,~Element,read(source,context,key),source => Traversal.collect_list(~&2,~Source,~Element,~Key,~Context,~read,rest,source,context),
          source => collect_list_preserves_source(~Source,~Element,~Key,~Context,~read,~kept,rest,source,context)),kept(source,context,key))

def collect_list_read(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,result: Source & Model) -> Reader & Runtime:
  (source,value) = result
  (pack_reader(source),pack_value(value))

# The loops gather newest first and reverse once; a gathered prefix stands
# in front of the recursive collection of the rest.
def onto(~Source: Data,~Model: Data,gathered: List<&2,Model>,result: Source & List<&2,Model>) -> Source & List<&2,Model>:
  (+source,+values) = result
  (source,Traversal.reversed_onto(~&2,~Model,gathered,values))

def onto_empty(~Source: Data,~Model: Data,result: Source & List<&2,Model>) -> {onto(~Source,~Model,[],result) == result : Source & List<&2,Model>}:
  (+source,+values) = result
  {==}

def onto_prepend(~Source: Data,~Model: Data,+value: Model,+gathered: List<&2,Model>,result: Source & List<&2,Model>) ->
  {onto(~Source,~Model,value <> gathered,result) == onto(~Source,~Model,gathered,Traversal.collect_prepend(~&2,~Source,~Model,value,result)) : Source & List<&2,Model>}:
  (+source,+values) = result
  {==}

def collect_values_reversed(~ownership: Quant,~Runtime: Kind(ownership),~Model: Data,~pack_value: Model -> Runtime,values: List<&2,Model>,+tail: List<&2,Model>) ->
  {collect_values(~ownership,~Runtime,~Model,~pack_value,Traversal.reversed_onto(~&2,~Model,values,tail)) ==
    Traversal.reversed_onto(~ownership,~Runtime,collect_values(~ownership,~Runtime,~Model,~pack_value,values),collect_values(~ownership,~Runtime,~Model,~pack_value,tail)) : List<ownership,Runtime>}:
  match values:
    case Nil{}: {==}
    case +head <> +rest: collect_values_reversed(~ownership,~Runtime,~Model,~pack_value,rest,head <> tail)

def gathering_finished(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,+source: Source,+gathered: List<&2,Model>) ->
  {(pack_reader(source),Traversal.reversed_onto(~ownership,~Runtime,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered),[])) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,(source,[]))) : Reader & List<ownership,Runtime>}:
  Equal.cong(List<ownership,Runtime>,Reader & List<ownership,Runtime>,values => (pack_reader(source),values),
    Traversal.reversed_onto(~ownership,~Runtime,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered),[]),
    collect_values(~ownership,~Runtime,~Model,~pack_value,Traversal.reversed_onto(~&2,~Model,gathered,[])),
    Equal.sym(List<ownership,Runtime>,collect_values(~ownership,~Runtime,~Model,~pack_value,Traversal.reversed_onto(~&2,~Model,gathered,[])),
      Traversal.reversed_onto(~ownership,~Runtime,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered),[]),
      collect_values_reversed(~ownership,~Runtime,~Model,~pack_value,gathered,[])))

def gathered_key_matches(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Key: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,head: Source & Model,+rest: List<&2,Key>,+context: Context,+gathered: List<&2,Model>,
  -runtime_rest: Traversal.Gathering<&1,ownership,Reader,Runtime,Traversal.Keyed<Key,Context>> -> Reader & List<ownership,Runtime>,model_rest: Source -> Source & List<&2,Model>,
  rest_matches: @+source: Source -> @+gathered: List<&2,Model> ->
    {runtime_rest(Traversal.Gathering{pack_reader(source),Traversal.Keyed{rest,context},collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)}) ==
      collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,model_rest(source))) : Reader & List<ownership,Runtime>}) ->
  {runtime_rest(Traversal.gathered_key(~ownership,~Reader,~Runtime,~Key,~Context,collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,head),rest,context,
      collect_values(~ownership,~Runtime,~Model,~pack_value,gathered))) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_list_next(~&2,~Source,~Model,head,model_rest))) : Reader & List<ownership,Runtime>}:
  (+source,+value) = head
  Equal.trans(Reader & List<ownership,Runtime>,
    runtime_rest(Traversal.Gathering{pack_reader(source),Traversal.Keyed{rest,context},collect_values(~ownership,~Runtime,~Model,~pack_value,value <> gathered)}),
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,value <> gathered,model_rest(source))),
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_prepend(~&2,~Source,~Model,value,model_rest(source)))),
    rest_matches(source,value <> gathered),
    Equal.cong(Source & List<&2,Model>,Reader & List<ownership,Runtime>,result => collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,result),
      onto(~Source,~Model,value <> gathered,model_rest(source)),onto(~Source,~Model,gathered,Traversal.collect_prepend(~&2,~Source,~Model,value,model_rest(source))),
      onto_prepend(~Source,~Model,value,gathered,model_rest(source))))

def collect_list_loop_gathers(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Key: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Key -> Reader & Runtime,~model_read: Source -> Context -> Key -> Source & Model,
  ~read_matches: @+source: Source -> @+context: Context -> @+key: Key ->
    {read(pack_reader(source),context,key) == collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key)) : Reader & Runtime},
  keys: List<&2,Key>,+source: Source,+context: Context,+gathered: List<&2,Model>) ->
  {Traversal.gathering_result(~ownership,~Reader,~Runtime,~Traversal.Keyed<Key,Context>,
      Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Traversal.Keyed<Key,Context>>,~Traversal.gather_key_step(~ownership,~Reader,~Runtime,~Key,~Context,~read),List.length(&2,Key,keys),
        Traversal.Gathering{pack_reader(source),Traversal.Keyed{keys,context},collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)})) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,keys,source,context))) : Reader & List<ownership,Runtime>}:
  match keys:
    case Nil{}: gathering_finished(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,source,gathered)
    case +key <> +rest:
      %Equal.sym(Reader & Runtime,read(pack_reader(source),context,key),collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key)),read_matches(source,context,key)) :
        {Traversal.gathering_result(~ownership,~Reader,~Runtime,~Traversal.Keyed<Key,Context>,
            Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Traversal.Keyed<Key,Context>>,~Traversal.gather_key_step(~ownership,~Reader,~Runtime,~Key,~Context,~read),List.length(&2,Key,rest),
              Traversal.gathered_key(~ownership,~Reader,~Runtime,~Key,~Context,_,rest,context,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)))) ==
          collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,key <> rest,source,context))) : Reader & List<ownership,Runtime>}
      gathered_key_matches(~ownership,~Reader,~Source,~Runtime,~Model,~Key,~Context,~pack_reader,~pack_value,model_read(source,context,key),rest,context,gathered,
        state => Traversal.gathering_result(~ownership,~Reader,~Runtime,~Traversal.Keyed<Key,Context>,
          Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Traversal.Keyed<Key,Context>>,~Traversal.gather_key_step(~ownership,~Reader,~Runtime,~Key,~Context,~read),List.length(&2,Key,rest),state)),
        source => Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,rest,source,context),
        source => gathered => collect_list_loop_gathers(~ownership,~Reader,~Source,~Runtime,~Model,~Key,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,rest,source,context,gathered))

# The keyed collection loop refines the recursive model collection.
def collect_list_loop_refines_model(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Key: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Key -> Reader & Runtime,~model_read: Source -> Context -> Key -> Source & Model,
  ~read_matches: @+source: Source -> @+context: Context -> @+key: Key ->
    {read(pack_reader(source),context,key) == collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key)) : Reader & Runtime},
  +keys: List<&2,Key>,+source: Source,+context: Context) ->
  {Traversal.collect_list_loop(~ownership,~Reader,~Runtime,~Key,~Context,~read,keys,pack_reader(source),context) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,keys,source,context)) : Reader & List<ownership,Runtime>}:
  %onto_empty(~Source,~Model,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,keys,source,context)) :
    {Traversal.collect_list_loop(~ownership,~Reader,~Runtime,~Key,~Context,~read,keys,pack_reader(source),context) ==
      collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,_) : Reader & List<ownership,Runtime>}
  collect_list_loop_gathers(~ownership,~Reader,~Source,~Runtime,~Model,~Key,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,keys,source,context,[])

def represent(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&1,&1,Reader,Array<Element>,Context>:
  Traversal.State{reader,output,position,context} = state
  Traversal.State{pack_reader(reader),Tree.pack(Element,output),position,context}

def model_step(~Model: Data,~Element: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{+reader,output,+position,+context} = state
  Traversal.State{reader,Tree.write(Element,output,destination(context,position),sample(reader,context,position)),(position + 1 : U32),context}

def model_reader(~Model: Data,~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Model:
  Traversal.State{reader,output,position,context} = state
  reader

def model_reader_step(~Model: Data,~Element: Data,~Context: Data,~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {model_reader(~Model,~Element,~Context,model_step(~Model,~Element,~Context,~sample,~destination,state)) == model_reader(~Model,~Element,~Context,state) : Model}:
  Traversal.State{reader,output,position,context} = state
  {==}

def model_reader_kept(~Model: Data,~Element: Data,~Context: Data,~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {model_reader(~Model,~Element,~Context,Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~model_step(~Model,~Element,~Context,~sample,~destination),count,state)) == model_reader(~Model,~Element,~Context,state) : Model}:
  Projection.preserves_observation(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~Model,~model_step(~Model,~Element,~Context,~sample,~destination),~model_reader(~Model,~Element,~Context),~model_reader_step(~Model,~Element,~Context,~sample,~destination),count,state)

def step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.step(~Reader,~Element,~Context,~read,~destination,represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+context} = state
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.store(~Reader,~Element,~Context,~destination,position,context,Tree.pack(Element,output),_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),destination(context,position),sample(reader,context,position)),
    Tree.pack(Element,Tree.write(Element,output,destination(context,position),sample(reader,context,position))),
    StorageLaws.write(Element,output,destination(context,position),sample(reader,context,position))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),context} ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_step(~Model,~Element,~Context,~sample,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def traversal_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.run(~Reader,~Element,~Context,~read,~destination,count,represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,~model_step(~Model,~Element,~Context,~sample,~destination),count,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
    ~Traversal.step(~Reader,~Element,~Context,~read,~destination),~model_step(~Model,~Element,~Context,~sample,~destination),
    ~represent(~Reader,~Model,~Element,~Context,~pack_reader),~step_refines_model(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~destination,~read_matches),count,state)
# Concrete copy/zip readers discharge the generic traversal's only read premise.

def pack_tree(~Element: Data,tree: Tree.Tree<Element>) -> Array<Element>:
  Tree.pack(Element,tree)

def sample_offset(~Element: Data,tree: Tree.Tree<Element>,context: Traversal.Offsets,position: U32) -> Element:
  Traversal.Offsets{source,destination} = context
  Tree.read(Element,tree,(source + position : U32))

def offset_read_refines_storage(~Element: Data,+tree: Tree.Tree<Element>,+context: Traversal.Offsets,+position: U32) ->
  {Traversal.source_offset(~Element,pack_tree(~Element,tree),context,position) ==
    (pack_tree(~Element,tree),sample_offset(~Element,tree,context,position)) : Array<Element> & Element}:
  Traversal.Offsets{source,destination} = context
  StorageLaws.read(Element,tree,(source + position : U32))

def sample_strided(~Element: Data,tree: Tree.Tree<Element>,context: Traversal.Strided,position: U32) -> Element:
  Traversal.Strided{base,stride} = context
  Tree.read(Element,tree,(base + position * stride : U32))

def strided_read_refines_storage(~Element: Data,+tree: Tree.Tree<Element>,+context: Traversal.Strided,+position: U32) ->
  {Traversal.strided(~Element,pack_tree(~Element,tree),context,position) ==
    (pack_tree(~Element,tree),sample_strided(~Element,tree,context,position)) : Array<Element> & Element}:
  Traversal.Strided{base,stride} = context
  StorageLaws.read(Element,tree,(base + position * stride : U32))

def copy_traversal_refines_storage(~Element: Data,+count: Nat,
  +state: Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>) ->
  {Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,count,
      represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),state)) ==
    represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
        ~model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~sample_offset(~Element),~Traversal.destination_offset),count,state)) :
    Traversal.State<&1,&1,Array<Element>,Array<Element>,Traversal.Offsets>}:
  traversal_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~pack_tree(~Element),
    ~Traversal.source_offset(~Element),~sample_offset(~Element),~Traversal.destination_offset,~offset_read_refines_storage(~Element),count,state)

# The accumulator is an observation, independent of its numeric interpretation.
# In particular this theorem never rewrites or reassociates a floating sum.
def represent_fold(~Reader: Type,~Model: Data,~Accumulator: Data,~Context: Data,~pack_reader: Model -> Reader,
  state: Traversal.State<&2,&2,Model,Accumulator,Context>) -> Traversal.State<&1,&2,Reader,Accumulator,Context>:
  Traversal.State{reader,accumulator,position,context} = state
  Traversal.State{pack_reader(reader),accumulator,position,context}

def model_fold_step(~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  state: Traversal.State<&2,&2,Model,Accumulator,Context>) -> Traversal.State<&2,&2,Model,Accumulator,Context>:
  Traversal.State{+reader,accumulator,+position,+context} = state
  Traversal.State{reader,combine(accumulator,sample(reader,context,position)),(position + 1 : U32),context}

def fold_step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine,state)) : Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  Traversal.State{+reader,+accumulator,+position,+context} = state
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),
    read_matches(reader,context,position)) :
    {Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,_) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
        model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine,Traversal.State{reader,accumulator,position,context})) :
      Traversal.State<&1,&2,Reader,Accumulator,Context>}
  {==}

def fold_refines_ordered_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),
      count,represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),count,state)) :
    Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.State<&2,&2,Model,Accumulator,Context>,
    ~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),
    ~represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader),
    ~fold_step_refines_model(~Reader,~Model,~Element,~Accumulator,~Context,~pack_reader,~read,~sample,~combine,~read_matches),count,state)

# In-place transforms retain one owner and use the same generic loop lifting.
def represent_map(~Element: Data,~Context: Data,
  state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&1,Unit,Array<Element>,Context>:
  Traversal.State{Unit{},output,position,context} = state
  Traversal.State{Unit{},Tree.pack(Element,output),position,context}

def unindexed(~Element: Data,~Context: Data,~map: Context -> Element -> Element,context: Context,position: U32,value: Element) -> Element:
  map(context,value)

def model_map_step(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,
  state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>:
  Traversal.State{Unit{},+output,+position,+context} = state
  Traversal.State{Unit{},Tree.write(Element,output,position,map(context,position,Tree.read(Element,output,position))),
    (position + 1 : U32),context}

def map_step_refines_model(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) ->
  {Traversal.map_step(~Element,~Context,~map,represent_map(~Element,~Context,state)) ==
    represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}:
  Traversal.State{Unit{},+output,+position,+context} = state
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),
    (Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.mapped_value(~Element,~Context,~map,position,context,_) ==
      represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,map(context,Tree.read(Element,output,position))),
    Tree.pack(Element,Tree.write(Element,output,position,map(context,Tree.read(Element,output,position)))),
    StorageLaws.write(Element,output,position,map(context,Tree.read(Element,output,position)))) :
    {Traversal.State{Unit{},_,(position + 1 : U32),context} ==
      represent_map(~Element,~Context,model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map),state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}
  {==}

def map_loop_refines_model(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +count: Nat,+state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) ->
  {Loop.run(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.map_step(~Element,~Context,~map),
      count,represent_map(~Element,~Context,state)) ==
    represent_map(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,
      ~model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map)),count,state)) : Traversal.State<&2,&1,Unit,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,
    ~Traversal.map_step(~Element,~Context,~map),~model_map_step(~Element,~Context,~unindexed(~Element,~Context,~map)),~represent_map(~Element,~Context),
    ~map_step_refines_model(~Element,~Context,~map),count,state)

# One representation proof covers the entire indexed output traversal. The
# strategy supplies a single-step result, not another outer-loop induction.

# In-place binary updates use the current destination value in left position.
# This theorem models sequential writes, including modular/overlapping raw
# addresses. Valid view guards and non-overlap are separate obligations.
def model_update_step(~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) -> Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>:
  Traversal.State{+reader,+output,+position,+context} = state
  Traversal.State{reader,Tree.write(Element,output,destination(context,position),
    combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position))),(position + 1 : U32),context}

def update_step_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Sample,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Sample},
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.update_step(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,
      represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Traversal.State{+reader,+output,+position,+context} = state
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),destination(context,position)),
    (Tree.pack(Element,output),Tree.read(Element,output,destination(context,position))),StorageLaws.read(Element,output,destination(context,position))) :
    {Traversal.update_current(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,pack_reader(reader),position,context,_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Reader & Sample,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.update_sample(~Reader,~Element,~Sample,~Context,~combine,~destination,position,context,Tree.pack(Element,output),
      Tree.read(Element,output,destination(context,position)),_) ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),destination(context,position),
      combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position))),
    Tree.pack(Element,Tree.write(Element,output,destination(context,position),
      combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position)))),
    StorageLaws.write(Element,output,destination(context,position),combine(Tree.read(Element,output,destination(context,position)),sample(reader,context,position)))) :
    {Traversal.State{pack_reader(reader),_,(position + 1 : U32),context} ==
      represent(~Reader,~Model,~Element,~Context,~pack_reader,model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination,
        Traversal.State{reader,output,position,context})) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}
  {==}

def update_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Sample: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Sample,
  ~sample: Model -> Context -> U32 -> Sample,~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Sample},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.update(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,count,
      represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    represent(~Reader,~Model,~Element,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
        ~model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),count,state)) :
    Traversal.State<&1,&1,Reader,Array<Element>,Context>}:
  Projection.iteration_refines_model(~Traversal.State<&1,&1,Reader,Array<Element>,Context>,~Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>,
    ~Traversal.update_step(~Reader,~Element,~Sample,~Context,~read,~combine,~destination),
    ~model_update_step(~Model,~Element,~Sample,~Context,~sample,~combine,~destination),
    ~represent(~Reader,~Model,~Element,~Context,~pack_reader),
    ~update_step_refines_model(~Reader,~Model,~Element,~Sample,~Context,~pack_reader,~read,~sample,~combine,~destination,~read_matches),count,state)

# A state with a moving counter is the same indexed fold as its stores.
def indexed_loop(~Runtime: Type,~Model: Data,~Context: Data,
  ~step: Runtime -> Runtime,~update: U32 -> Context -> Model -> Model,
  ~represent: Context -> U32 -> Model -> Runtime,
  ~step_matches: @+context: Context -> @+index: U32 -> @+state: Model ->
    {step(represent(context,index,state)) == represent(context,(index + 1 : U32),update(index,context,state)) : Runtime},
  count: Nat,+index: U32,+context: Context,+state: Model) ->
  {Loop.run(~Runtime,~step,count,represent(context,index,state)) ==
    represent(context,(index + U32.from_nat(count) : U32),Traversal.indexed_fold(~Model,~Context,~update,count,index,context,state)) : Runtime}:
  match count:
    case 0n:
      %Equal.sym(U32,(index + 0 : U32),index,Arithmetic.add_zero(index)) :
        {represent(context,index,state) == represent(context,_,state) : Runtime}
      {==}
    case 1n+ +rest:
      %Equal.sym(Runtime,step(represent(context,index,state)),represent(context,(index + 1 : U32),update(index,context,state)),step_matches(context,index,state)) :
        {Loop.run(~Runtime,~step,rest,_) == represent(context,(index + U32.from_nat(1n+rest) : U32),
          Traversal.indexed_fold(~Model,~Context,~update,rest,(index + 1 : U32),context,update(index,context,state))) : Runtime}
      %Equal.sym(U32,(index + U32.from_nat(1n+rest) : U32),((index + 1 : U32) + U32.from_nat(rest) : U32),Projection.indexed_successor(index,rest)) :
        {Loop.run(~Runtime,~step,rest,represent(context,(index + 1 : U32),update(index,context,state))) ==
          represent(context,_,Traversal.indexed_fold(~Model,~Context,~update,rest,(index + 1 : U32),context,update(index,context,state))) : Runtime}
      indexed_loop(~Runtime,~Model,~Context,~step,~update,~represent,~step_matches,rest,(index + 1 : U32),context,update(index,context,state))

def indexed_fold_refines_model(~Runtime: Type,~Model: Data,~Context: Data,
  ~update: U32 -> Context -> Runtime -> Runtime,~model_update: U32 -> Context -> Model -> Model,
  ~represent: Model -> Runtime,
  ~step_matches: @+index: U32 -> @+context: Context -> @+state: Model ->
    {update(index,context,represent(state)) == represent(model_update(index,context,state)) : Runtime},
  +count: Nat,+index: U32,+context: Context,+state: Model) ->
  {Traversal.indexed_fold(~Runtime,~Context,~update,count,index,context,represent(state)) ==
    represent(Traversal.indexed_fold(~Model,~Context,~model_update,count,index,context,state)) : Runtime}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Runtime,update(index,context,represent(state)),represent(model_update(index,context,state)),step_matches(index,context,state)) :
        {Traversal.indexed_fold(~Runtime,~Context,~update,rest,(index + 1 : U32),context,_) ==
          represent(Traversal.indexed_fold(~Model,~Context,~model_update,rest,(index + 1 : U32),context,model_update(index,context,state))) : Runtime}
      indexed_fold_refines_model(~Runtime,~Model,~Context,~update,~model_update,~represent,~step_matches,
        rest,(index + 1 : U32),context,model_update(index,context,state))

# An indexed traversal transports a fixed observation through its own update.
def indexed_fold_observes(~State: Data,~Observation: Data,~Context: Data,
  ~update: U32 -> Context -> State -> State,~observe: State -> Observation,
  ~observed_update: U32 -> Context -> Observation -> Observation,
  ~step_observed: @+index: U32 -> @+context: Context -> @+state: State ->
    {observe(update(index,context,state)) == observed_update(index,context,observe(state)) : Observation},
  count: Nat,+first: U32,+context: Context,+state: State) ->
  {observe(Traversal.indexed_fold(~State,~Context,~update,count,first,context,state)) ==
    Traversal.indexed_fold(~Observation,~Context,~observed_update,count,first,context,observe(state)) : Observation}:
  match count:
    case 0n: {==}
    case 1n+rest:
      %step_observed(first,context,state) :
        {observe(Traversal.indexed_fold(~State,~Context,~update,rest,(first + 1 : U32),context,update(first,context,state))) ==
          Traversal.indexed_fold(~Observation,~Context,~observed_update,rest,(first + 1 : U32),context,_) : Observation}
      indexed_fold_observes(~State,~Observation,~Context,~update,~observe,~observed_update,~step_observed,
        rest,(first + 1 : U32),context,update(first,context,state))

def indexed_fold_fixed(~State: Data,~Context: Data,~update: U32 -> Context -> State -> State,
  ~fixed: @+index: U32 -> @+context: Context -> @+state: State -> {update(index,context,state) == state : State},
  count: Nat,+first: U32,+context: Context,+state: State) ->
  {Traversal.indexed_fold(~State,~Context,~update,count,first,context,state) == state : State}:
  match count:
    case 0n: {==}
    case 1n+rest:
      %Equal.sym(State,update(first,context,state),state,fixed(first,context,state)) :
        {Traversal.indexed_fold(~State,~Context,~update,rest,(first + 1 : U32),context,_) == state : State}
      indexed_fold_fixed(~State,~Context,~update,~fixed,rest,(first + 1 : U32),context,state)

# One bounded induction carries capacity, preserved sources and completed
# output values together. A strategy proves its local update obligation;
# it does not reimplement the whole output traversal proof.
def indexed_fold_change_context(~State: Data,~Outer: Data,~Inner: Data,
  ~project: Outer -> Inner,~outer_step: U32 -> Outer -> State -> State,~inner_step: U32 -> Inner -> State -> State,
  ~same_step: @+index: U32 -> @+context: Outer -> @+state: State ->
    {outer_step(index,context,state) == inner_step(index,project(context),state) : State},
  +count: Nat,+first: U32,+context: Outer,+state: State) ->
  {Traversal.indexed_fold(~State,~Outer,~outer_step,count,first,context,state) ==
    Traversal.indexed_fold(~State,~Inner,~inner_step,count,first,project(context),state) : State}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(State,outer_step(first,context,state),inner_step(first,project(context),state),same_step(first,context,state)) :
        {Traversal.indexed_fold(~State,~Outer,~outer_step,rest,(first + 1 : U32),context,_) ==
          Traversal.indexed_fold(~State,~Inner,~inner_step,rest,(first + 1 : U32),project(context),inner_step(first,project(context),state)) : State}
      indexed_fold_change_context(~State,~Outer,~Inner,~project,~outer_step,~inner_step,~same_step,
        rest,(first + 1 : U32),context,inner_step(first,project(context),state))

def indexed_fold_preserves_invariant(~State: Data,~Context: Data,
  ~limit: Context -> Nat,~Invariant: Context -> Nat -> State -> Data,
  ~update: U32 -> Context -> State -> State,
  ~preserve: @+context: Context -> @+index: Nat -> @+state: State ->
    {Nat.is_lt(index,limit(context)) == True{} : Bool} -> Invariant(context,index,state) ->
    Invariant(context,1n+index,update(U32.from_nat(index),context,state)),
  +count: Nat,+first: Nat,+context: Context,+state: State,
  +room: {Nat.is_le(Nat.add(first,count),limit(context)) == True{} : Bool},
  evidence: Invariant(context,first,state)) ->
  Invariant(context,Nat.add(first,count),Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state)):
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : Invariant(context,_,state)
      evidence
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        Invariant(context,Nat.add(first,1n+rest),Traversal.indexed_fold(~State,~Context,~update,rest,_,context,update(U32.from_nat(first),context,state)))
      %Equal.sym(Nat,Nat.add(first,1n+rest),1n+Nat.add(first,rest),Addition.successor_right(first,rest)) :
        Invariant(context,_,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+first),context,update(U32.from_nat(first),context,state)))
      indexed_fold_preserves_invariant(~State,~Context,~limit,~Invariant,~update,~preserve,rest,1n+first,context,update(U32.from_nat(first),context,state),
        Order.advance_room(first,rest,limit(context),room),preserve(context,first,state,Order.room_index(first,rest,limit(context),room),evidence))

# Splitting a traversal preserves every state update in the same order. The
# boundary uses modular U32 conversion; no scalar/FP32-specific induction.
def indexed_fold_split(~Value: Type,~Context: Data,~update: U32 -> Context -> Value -> Value,
  +first: Nat,+second: Nat,+start: Nat,+context: Context,-initial: Value) ->
  {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(first,second),U32.from_nat(start),context,initial) ==
    Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(Nat.add(start,first)),context,
      Traversal.indexed_fold(~Value,~Context,~update,first,U32.from_nat(start),context,initial)) : Value}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(start),context,initial) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(_),context,initial) : Value}
      {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(rest,second),_,context,update(U32.from_nat(start),context,initial)) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(Nat.add(start,1n+rest)),context,
            Traversal.indexed_fold(~Value,~Context,~update,rest,_,context,update(U32.from_nat(start),context,initial))) : Value}
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Traversal.indexed_fold(~Value,~Context,~update,Nat.add(rest,second),U32.from_nat(1n+start),context,update(U32.from_nat(start),context,initial)) ==
          Traversal.indexed_fold(~Value,~Context,~update,second,U32.from_nat(_),context,
            Traversal.indexed_fold(~Value,~Context,~update,rest,U32.from_nat(1n+start),context,update(U32.from_nat(start),context,initial))) : Value}
      indexed_fold_split(~Value,~Context,~update,rest,second,1n+start,context,update(U32.from_nat(start),context,initial))

# One leaf supplies source, value and observation facts. The shared collection
# lifts all three together, including leaf premises for each input key.
def collected_leaf_frame(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,context: Context,key: Key,result: Source & Element) -> Data:
  (+source,+value) = result
  KernelGemm.Both<Frame(context,source),KernelGemm.Both<ValueFrame(context,value),{observe(context,value) == sample(context,key) : Model}>>

def collected_list_frame(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,+context: Context,+keys: List<&2,Key>,result: Source & List<&2,Element>) -> Data:
  (+source,+values) = result
  KernelGemm.Both<Frame(context,source),KernelGemm.Both<Members.All(Element,{value => ValueFrame(context,value) : Element -> Data},values),
    {collect_values_context(~&2,~Model,~Element,~Context,~observe,values,context) ==
      collect_values_context(~&2,~Model,~Key,~Context,~sample,keys,context) : List<&2,Model>}>>

def collect_list_certified_join(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,+context: Context,+key: Key,+rest: List<&2,Key>,+value: Element,
  tail: Source & List<&2,Element>,value_frame: ValueFrame(context,value),+value_equal: {observe(context,value) == sample(context,key) : Model},
  tail_frame: collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,rest,tail)) ->
  collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key <> rest,Traversal.collect_prepend(~&2,~Source,~Element,value,tail)):
  (source,+values) = tail
  KernelGemm.Both{source_frame,KernelGemm.Both{values_frame,+values_equal}} = tail_frame
  KernelGemm.Both{source_frame,KernelGemm.Both{KernelGemm.Both{value_frame,values_frame},
    %value_equal : {observe(context,value) <> collect_values_context(~&2,~Model,~Element,~Context,~observe,values,context) == _ <> collect_values_context(~&2,~Model,~Key,~Context,~sample,rest,context) : List<&2,Model>}
    Equal.cong(List<&2,Model>,List<&2,Model>,values => observe(context,value) <> values,
      collect_values_context(~&2,~Model,~Element,~Context,~observe,values,context),
      collect_values_context(~&2,~Model,~Key,~Context,~sample,rest,context),values_equal)}}

def collect_list_certified_next(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,+context: Context,+key: Key,+rest: List<&2,Key>,
  head: Source & Element,tail: Source -> Source & List<&2,Element>,
  head_frame: collected_leaf_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key,head),
  tail_frame: @+source: Source -> Frame(context,source) -> collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,rest,tail(source))) ->
  collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key <> rest,
    Traversal.collect_list_next(~&2,~Source,~Element,head,tail)):
  (+source,+value) = head
  KernelGemm.Both{source_frame,KernelGemm.Both{value_frame,+value_equal}} = head_frame
  collect_list_certified_join(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key,rest,value,tail(source),value_frame,value_equal,tail_frame(source,source_frame))

def collect_list_certified(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,~ReadContext: Data,~project: Context -> ReadContext,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,~KeyFrame: Context -> Key -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,
  ~read: Source -> ReadContext -> Key -> Source & Element,
  ~leaf: @+source: Source -> @+context: Context -> @+key: Key -> Frame(context,source) -> KeyFrame(context,key) ->
    collected_leaf_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key,read(source,project(context),key)),
  +keys: List<&2,Key>,+source: Source,+context: Context,source_frame: Frame(context,source),
  keys_frame: Members.All(Key,{key => KeyFrame(context,key) : Key -> Data},keys)) ->
  collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,keys,
    Traversal.collect_list(~&2,~Source,~Element,~Key,~ReadContext,~read,keys,source,project(context))):
  match keys:
    case Nil{}: KernelGemm.Both{source_frame,KernelGemm.Both{Unit{},{==}}}
    case +key <> +rest:
      KernelGemm.Both{head_frame,+tail_frame} = keys_frame
      collect_list_certified_next(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key,rest,read(source,project(context),key),
        source => Traversal.collect_list(~&2,~Source,~Element,~Key,~ReadContext,~read,rest,source,project(context)),leaf(source,context,key,source_frame,head_frame),
        source => source_frame => collect_list_certified(~Source,~Element,~Key,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~KeyFrame,~observe,~sample,~read,~leaf,rest,source,context,source_frame,tail_frame))

def indexed_fold_change_context_under(~State: Data,~Outer: Data,~Inner: Data,
  ~project: Outer -> Inner,~Condition: Outer -> Data,
  ~outer_step: U32 -> Outer -> State -> State,~inner_step: U32 -> Inner -> State -> State,
  ~same_step: @+index: U32 -> @+context: Outer -> @+state: State -> Condition(context) ->
    {outer_step(index,context,state) == inner_step(index,project(context),state) : State},
  count: Nat,+first: U32,+context: Outer,+state: State,+condition: Condition(context)) ->
  {Traversal.indexed_fold(~State,~Outer,~outer_step,count,first,context,state) ==
    Traversal.indexed_fold(~State,~Inner,~inner_step,count,first,project(context),state) : State}:
  match count:
    case 0n: {==}
    case 1n+rest:
      %Equal.sym(State,outer_step(first,context,state),inner_step(first,project(context),state),same_step(first,context,state,condition)) :
        {Traversal.indexed_fold(~State,~Outer,~outer_step,rest,(first + 1 : U32),context,_) ==
          Traversal.indexed_fold(~State,~Inner,~inner_step,rest,(first + 1 : U32),project(context),inner_step(first,project(context),state)) : State}
      indexed_fold_change_context_under(~State,~Outer,~Inner,~project,~Condition,~outer_step,~inner_step,~same_step,rest,(first + 1 : U32),context,inner_step(first,project(context),state),condition)
