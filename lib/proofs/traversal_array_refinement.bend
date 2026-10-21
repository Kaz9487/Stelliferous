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
import ./storage_shape.bend as Shape
import ./ordered_ranges.bend as Ranges
import ./list_storage_readback.bend as ListRead
import ./bounded_u32_arithmetic.bend as Arithmetic

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

def collected_result(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,result: Source & Traversal.Collected<&2,Context,Model>) -> Reader & Traversal.Collected<ownership,Context,Runtime>:
  (source,Traversal.Collected{context,value}) = result
  (pack_reader(source),Traversal.Collected{context,pack_value(value)})

def collect_prepend_matches(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,+value: Model,result: Source & List<&2,Model>) ->
  {Traversal.collect_prepend(~ownership,~Reader,~Runtime,pack_value(value),collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,result)) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_prepend(~&2,~Source,~Model,value,result)) : Reader & List<ownership,Runtime>}:
  (+source,+values) = result
  {==}

def collect_next_matches(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,head: Source & Traversal.Collected<&2,Context,Model>,
  -runtime_tail: Reader -> Context -> Reader & List<ownership,Runtime>,model_tail: Source -> Context -> Source & List<&2,Model>,
  tail_matches: @+source: Source -> @+context: Context ->
    {runtime_tail(pack_reader(source),context) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_tail(source,context)) : Reader & List<ownership,Runtime>}) ->
  {Traversal.collect_next(~ownership,~Reader,~Runtime,~Context,collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,head),runtime_tail) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_next(~&2,~Source,~Model,~Context,head,model_tail)) : Reader & List<ownership,Runtime>}:
  (+source,Traversal.Collected{+context,+value}) = head
  %Equal.sym(Reader & List<ownership,Runtime>,runtime_tail(pack_reader(source),context),collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_tail(source,context)),tail_matches(source,context)) :
    {Traversal.collect_prepend(~ownership,~Reader,~Runtime,pack_value(value),_) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_prepend(~&2,~Source,~Model,value,model_tail(source,context))) : Reader & List<ownership,Runtime>}
  collect_prepend_matches(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,value,model_tail(source,context))

def collect_refines_model(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Reader & Traversal.Collected<ownership,Context,Runtime>,~model_read: Source -> Context -> Source & Traversal.Collected<&2,Context,Model>,
  ~read_matches: @+source: Source -> @+context: Context ->
    {read(pack_reader(source),context) == collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context)) : Reader & Traversal.Collected<ownership,Context,Runtime>},
  count: Nat,+source: Source,+context: Context) ->
  {Traversal.collect(~ownership,~Reader,~Runtime,~Context,~read,count,pack_reader(source),context) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,count,source,context)) : Reader & List<ownership,Runtime>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Reader & Traversal.Collected<ownership,Context,Runtime>,read(pack_reader(source),context),collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context)),read_matches(source,context)) :
        {Traversal.collect_next(~ownership,~Reader,~Runtime,~Context,_,reader => context => Traversal.collect(~ownership,~Reader,~Runtime,~Context,~read,rest,reader,context)) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,1n+rest,source,context)) : Reader & List<ownership,Runtime>}
      collect_next_matches(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context),
        reader => context => Traversal.collect(~ownership,~Reader,~Runtime,~Context,~read,rest,reader,context),source => context => Traversal.collect(~&2,~Source,~Model,~Context,~model_read,rest,source,context),
        source => context => collect_refines_model(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,rest,source,context))

def collect_identity(~Element: Data,~Context: Data,context: Context,value: Element) -> Element:
  value

def collect_values_context_identity(~Element: Data,~Context: Data,values: List<&2,Element>,+context: Context) ->
  {collect_values_context(~&2,~Element,~Element,~Context,~collect_identity(~Element,~Context),values,context) == values : List<&2,Element>}:
  match values:
    case Nil{}: {==}
    case +head <> +rest:
      Equal.cong(List<&2,Element>,List<&2,Element>,values => head <> values,
        collect_values_context(~&2,~Element,~Element,~Context,~collect_identity(~Element,~Context),rest,context),rest,collect_values_context_identity(~Element,~Context,rest,context))

def collect_values_identity(~Element: Data,values: List<&2,Element>) ->
  {collect_values(~&2,~Element,~Element,~(value => value),values) == values : List<&2,Element>}:
  collect_values_context_identity(~Element,~Unit,values,Unit{})

def collect_result_identity(~Reader: Type,~Source: Data,~Element: Data,~pack_reader: Source -> Reader,result: Source & List<&2,Element>) ->
  {collect_result(~&2,~Reader,~Source,~Element,~Element,~pack_reader,~(value => value),result) ==
    (pack_reader(Pair.fst(Source,List<&2,Element>,result)),Pair.snd(Source,List<&2,Element>,result)) : Reader & List<&2,Element>}:
  (source,+values) = result
  Equal.cong(List<&2,Element>,Reader & List<&2,Element>,values => (pack_reader(source),values),collect_values(~&2,~Element,~Element,~(value => value),values),values,collect_values_identity(~Element,values))

def collect_prepend_source(~Source: Data,~Element: Data,+value: Element,result: Source & List<&2,Element>) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect_prepend(~&2,~Source,~Element,value,result)) == Pair.fst(Source,List<&2,Element>,result) : Source}:
  (source,values) = result
  {==}

def collect_next_source(~Source: Data,~Element: Data,~Context: Data,head: Source & Traversal.Collected<&2,Context,Element>,
  tail: Source -> Context -> Source & List<&2,Element>,
  kept: @+source: Source -> @+context: Context -> {Pair.fst(Source,List<&2,Element>,tail(source,context)) == source : Source}) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect_next(~&2,~Source,~Element,~Context,head,tail)) == Pair.fst(Source,Traversal.Collected<&2,Context,Element>,head) : Source}:
  (+source,+collected) = head
  Traversal.Collected{+context,+value} = collected
  Equal.trans(Source,Pair.fst(Source,List<&2,Element>,Traversal.collect_prepend(~&2,~Source,~Element,value,tail(source,context))),Pair.fst(Source,List<&2,Element>,tail(source,context)),source,collect_prepend_source(~Source,~Element,value,tail(source,context)),kept(source,context))

def collect_preserves_source(~Source: Data,~Element: Data,~Context: Data,
  ~read: Source -> Context -> Source & Traversal.Collected<&2,Context,Element>,
  ~kept: @+source: Source -> @+context: Context -> {Pair.fst(Source,Traversal.Collected<&2,Context,Element>,read(source,context)) == source : Source},
  count: Nat,+source: Source,+context: Context) ->
  {Pair.fst(Source,List<&2,Element>,Traversal.collect(~&2,~Source,~Element,~Context,~read,count,source,context)) == source : Source}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(Source,Pair.fst(Source,List<&2,Element>,Traversal.collect(~&2,~Source,~Element,~Context,~read,1n+rest,source,context)),Pair.fst(Source,Traversal.Collected<&2,Context,Element>,read(source,context)),source,
        collect_next_source(~Source,~Element,~Context,read(source,context),source => context => Traversal.collect(~&2,~Source,~Element,~Context,~read,rest,source,context),
          source => context => collect_preserves_source(~Source,~Element,~Context,~read,~kept,rest,source,context)),kept(source,context))

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

def collect_list_next_matches(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,head: Source & Model,
  -runtime_tail: Reader -> Reader & List<ownership,Runtime>,model_tail: Source -> Source & List<&2,Model>,
  tail_matches: @+source: Source ->
    {runtime_tail(pack_reader(source)) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_tail(source)) : Reader & List<ownership,Runtime>}) ->
  {Traversal.collect_list_next(~ownership,~Reader,~Runtime,collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,head),runtime_tail) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_list_next(~&2,~Source,~Model,head,model_tail)) : Reader & List<ownership,Runtime>}:
  (+source,+value) = head
  %Equal.sym(Reader & List<ownership,Runtime>,runtime_tail(pack_reader(source)),collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_tail(source)),tail_matches(source)) :
    {Traversal.collect_prepend(~ownership,~Reader,~Runtime,pack_value(value),_) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_prepend(~&2,~Source,~Model,value,model_tail(source))) : Reader & List<ownership,Runtime>}
  collect_prepend_matches(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,value,model_tail(source))

def collect_list_refines_model(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Key: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Key -> Reader & Runtime,~model_read: Source -> Context -> Key -> Source & Model,
  ~read_matches: @+source: Source -> @+context: Context -> @+key: Key ->
    {read(pack_reader(source),context,key) == collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key)) : Reader & Runtime},
  keys: List<&2,Key>,+source: Source,+context: Context) ->
  {Traversal.collect_list(~ownership,~Reader,~Runtime,~Key,~Context,~read,keys,pack_reader(source),context) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,keys,source,context)) : Reader & List<ownership,Runtime>}:
  match keys:
    case Nil{}: {==}
    case +key <> +rest:
      %Equal.sym(Reader & Runtime,read(pack_reader(source),context,key),collect_list_read(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key)),read_matches(source,context,key)) :
        {Traversal.collect_list_next(~ownership,~Reader,~Runtime,_,reader => Traversal.collect_list(~ownership,~Reader,~Runtime,~Key,~Context,~read,rest,reader,context)) == collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,keys,source,context)) : Reader & List<ownership,Runtime>}
      collect_list_next_matches(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,model_read(source,context,key),
        reader => Traversal.collect_list(~ownership,~Reader,~Runtime,~Key,~Context,~read,rest,reader,context),source => Traversal.collect_list(~&2,~Source,~Model,~Key,~Context,~model_read,rest,source,context),
        source => collect_list_refines_model(~ownership,~Reader,~Source,~Runtime,~Model,~Key,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,rest,source,context))

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

def gathered_matches(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,head: Source & Traversal.Collected<&2,Context,Model>,+gathered: List<&2,Model>,
  -runtime_rest: Traversal.Gathering<&1,ownership,Reader,Runtime,Context> -> Reader & List<ownership,Runtime>,model_rest: Source -> Context -> Source & List<&2,Model>,
  rest_matches: @+source: Source -> @+context: Context -> @+gathered: List<&2,Model> ->
    {runtime_rest(Traversal.Gathering{pack_reader(source),context,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)}) ==
      collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,model_rest(source,context))) : Reader & List<ownership,Runtime>}) ->
  {runtime_rest(Traversal.gathered(~ownership,~Reader,~Runtime,~Context,collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,head),
      collect_values(~ownership,~Runtime,~Model,~pack_value,gathered))) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_next(~&2,~Source,~Model,~Context,head,model_rest))) : Reader & List<ownership,Runtime>}:
  (+source,Traversal.Collected{+context,+value}) = head
  Equal.trans(Reader & List<ownership,Runtime>,
    runtime_rest(Traversal.Gathering{pack_reader(source),context,collect_values(~ownership,~Runtime,~Model,~pack_value,value <> gathered)}),
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,value <> gathered,model_rest(source,context))),
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect_prepend(~&2,~Source,~Model,value,model_rest(source,context)))),
    rest_matches(source,context,value <> gathered),
    Equal.cong(Source & List<&2,Model>,Reader & List<ownership,Runtime>,result => collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,result),
      onto(~Source,~Model,value <> gathered,model_rest(source,context)),onto(~Source,~Model,gathered,Traversal.collect_prepend(~&2,~Source,~Model,value,model_rest(source,context))),
      onto_prepend(~Source,~Model,value,gathered,model_rest(source,context))))

def collect_loop_gathers(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Reader & Traversal.Collected<ownership,Context,Runtime>,~model_read: Source -> Context -> Source & Traversal.Collected<&2,Context,Model>,
  ~read_matches: @+source: Source -> @+context: Context ->
    {read(pack_reader(source),context) == collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context)) : Reader & Traversal.Collected<ownership,Context,Runtime>},
  count: Nat,+source: Source,+context: Context,+gathered: List<&2,Model>) ->
  {Traversal.gathering_result(~ownership,~Reader,~Runtime,~Context,
      Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Context>,~Traversal.gather_step(~ownership,~Reader,~Runtime,~Context,~read),count,
        Traversal.Gathering{pack_reader(source),context,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)})) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,count,source,context))) : Reader & List<ownership,Runtime>}:
  match count:
    case 0n: gathering_finished(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,source,gathered)
    case 1n+ +rest:
      %Equal.sym(Reader & Traversal.Collected<ownership,Context,Runtime>,read(pack_reader(source),context),collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context)),read_matches(source,context)) :
        {Traversal.gathering_result(~ownership,~Reader,~Runtime,~Context,
            Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Context>,~Traversal.gather_step(~ownership,~Reader,~Runtime,~Context,~read),rest,
              Traversal.gathered(~ownership,~Reader,~Runtime,~Context,_,collect_values(~ownership,~Runtime,~Model,~pack_value,gathered)))) ==
          collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,onto(~Source,~Model,gathered,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,1n+rest,source,context))) : Reader & List<ownership,Runtime>}
      gathered_matches(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context),gathered,
        state => Traversal.gathering_result(~ownership,~Reader,~Runtime,~Context,
          Loop.run(~Traversal.Gathering<&1,ownership,Reader,Runtime,Context>,~Traversal.gather_step(~ownership,~Reader,~Runtime,~Context,~read),rest,state)),
        source => context => Traversal.collect(~&2,~Source,~Model,~Context,~model_read,rest,source,context),
        source => context => gathered => collect_loop_gathers(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,rest,source,context,gathered))

# The collection loop refines the recursive model collection.
def collect_loop_refines_model(~ownership: Quant,~Reader: Type,~Source: Data,~Runtime: Kind(ownership),~Model: Data,~Context: Data,
  ~pack_reader: Source -> Reader,~pack_value: Model -> Runtime,
  ~read: Reader -> Context -> Reader & Traversal.Collected<ownership,Context,Runtime>,~model_read: Source -> Context -> Source & Traversal.Collected<&2,Context,Model>,
  ~read_matches: @+source: Source -> @+context: Context ->
    {read(pack_reader(source),context) == collected_result(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,model_read(source,context)) : Reader & Traversal.Collected<ownership,Context,Runtime>},
  +count: Nat,+source: Source,+context: Context) ->
  {Traversal.collect_loop(~ownership,~Reader,~Runtime,~Context,~read,count,pack_reader(source),context) ==
    collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,count,source,context)) : Reader & List<ownership,Runtime>}:
  %onto_empty(~Source,~Model,Traversal.collect(~&2,~Source,~Model,~Context,~model_read,count,source,context)) :
    {Traversal.collect_loop(~ownership,~Reader,~Runtime,~Context,~read,count,pack_reader(source),context) ==
      collect_result(~ownership,~Reader,~Source,~Runtime,~Model,~pack_reader,~pack_value,_) : Reader & List<ownership,Runtime>}
  collect_loop_gathers(~ownership,~Reader,~Source,~Runtime,~Model,~Context,~pack_reader,~pack_value,~read,~model_read,~read_matches,count,source,context,[])

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

# The eight reads of a round are the samples at position, position + 1, ...
def read_last_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element,+c: Element,+d: Element,+e: Element,+f: Element,+g: Element) ->
  {Traversal.read_last(~Reader,~Element,a,b,c,d,e,f,g,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,c,d,e,f,g,sample(reader,context,position)}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_last(~Reader,~Element,a,b,c,d,e,f,g,_) == (pack_reader(reader),Traversal.Eight{a,b,c,d,e,f,g,sample(reader,context,position)}) : Reader & Traversal.Eight<Element>}
  {==}

def read_seventh_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element,+c: Element,+d: Element,+e: Element,+f: Element) ->
  {Traversal.read_seventh(~Reader,~Element,~Context,~read,position,context,a,b,c,d,e,f,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,c,d,e,f,sample(reader,context,position),sample(reader,context,(position + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_seventh(~Reader,~Element,~Context,~read,position,context,a,b,c,d,e,f,_) ==
      (pack_reader(reader),Traversal.Eight{a,b,c,d,e,f,sample(reader,context,position),sample(reader,context,(position + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_last_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,b,c,d,e,f,sample(reader,context,position))

def read_sixth_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element,+c: Element,+d: Element,+e: Element) ->
  {Traversal.read_sixth(~Reader,~Element,~Context,~read,position,context,a,b,c,d,e,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,c,d,e,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_sixth(~Reader,~Element,~Context,~read,position,context,a,b,c,d,e,_) ==
      (pack_reader(reader),Traversal.Eight{a,b,c,d,e,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
        sample(reader,context,((position + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_seventh_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,b,c,d,e,sample(reader,context,position))

def read_fifth_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element,+c: Element,+d: Element) ->
  {Traversal.read_fifth(~Reader,~Element,~Context,~read,position,context,a,b,c,d,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,c,d,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_fifth(~Reader,~Element,~Context,~read,position,context,a,b,c,d,_) ==
      (pack_reader(reader),Traversal.Eight{a,b,c,d,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
        sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_sixth_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,b,c,d,sample(reader,context,position))

def read_fourth_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element,+c: Element) ->
  {Traversal.read_fourth(~Reader,~Element,~Context,~read,position,context,a,b,c,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,c,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_fourth(~Reader,~Element,~Context,~read,position,context,a,b,c,_) ==
      (pack_reader(reader),Traversal.Eight{a,b,c,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
        sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_fifth_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,b,c,sample(reader,context,position))

def read_third_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element,+b: Element) ->
  {Traversal.read_third(~Reader,~Element,~Context,~read,position,context,a,b,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,b,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_third(~Reader,~Element,~Context,~read,position,context,a,b,_) ==
      (pack_reader(reader),Traversal.Eight{a,b,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
        sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_fourth_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,b,sample(reader,context,position))

def read_second_matches(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  ~read: Reader -> Context -> U32 -> Reader & Element,~sample: Model -> Context -> U32 -> Element,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +reader: Model,+context: Context,+position: U32,+a: Element) ->
  {Traversal.read_second(~Reader,~Element,~Context,~read,position,context,a,read(pack_reader(reader),context,position)) ==
    (pack_reader(reader),Traversal.Eight{a,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}:
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.read_second(~Reader,~Element,~Context,~read,position,context,a,_) ==
      (pack_reader(reader),Traversal.Eight{a,sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
        sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
        sample(reader,context,((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}) : Reader & Traversal.Eight<Element>}
  read_third_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),a,sample(reader,context,position))

# The model of eight fold steps.
def model_round(~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  state: Traversal.State<&2,&2,Model,Accumulator,Context>) -> Traversal.State<&2,&2,Model,Accumulator,Context>:
  Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),8n,state)

def round_refines_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~combine,
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine,state)) : Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  Traversal.State{+reader,+accumulator,+position,+context} = state
  %Equal.sym(Reader & Element,read(pack_reader(reader),context,position),(pack_reader(reader),sample(reader,context,position)),read_matches(reader,context,position)) :
    {Traversal.round_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,
      Traversal.read_eight(~Reader,~Element,~Context,~read,position,context,_)) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
        model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine,Traversal.State{reader,accumulator,position,context})) : Traversal.State<&1,&2,Reader,Accumulator,Context>}
  %Equal.sym(Reader & Traversal.Eight<Element>,
    Traversal.read_second(~Reader,~Element,~Context,~read,(position + 1 : U32),context,sample(reader,context,position),read(pack_reader(reader),context,(position + 1 : U32))),
    (pack_reader(reader),Traversal.Eight{sample(reader,context,position),sample(reader,context,(position + 1 : U32)),
      sample(reader,context,((position + 1 : U32) + 1 : U32)),sample(reader,context,(((position + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,(((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32)),
      sample(reader,context,(((((((position + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32) + 1 : U32))}),
    read_second_matches(~Reader,~Model,~Element,~Context,~pack_reader,~read,~sample,~read_matches,reader,context,(position + 1 : U32),sample(reader,context,position))) :
    {Traversal.round_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,_) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
        model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine,Traversal.State{reader,accumulator,position,context})) : Traversal.State<&1,&2,Reader,Accumulator,Context>}
  {==}

def round_steps(~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,-state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine,state) ==
    Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),8n,state) : Traversal.State<&2,&2,Model,Accumulator,Context>}:
  {==}

# fold_rounds runs the ordered model fold: count % 8 steps, then rounds of eight.
def rounds_refine_ordered_model(~Reader: Type,~Model: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack_reader: Model -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Model -> Context -> U32 -> Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_matches: @+reader: Model -> @+context: Context -> @+position: U32 ->
    {read(pack_reader(reader),context,position) == (pack_reader(reader),sample(reader,context,position)) : Reader & Element},
  +count: Nat,+state: Traversal.State<&2,&2,Model,Accumulator,Context>) ->
  {Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.div(count,8n),
      Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.mod(count,8n),
        represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state))) ==
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
      Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),count,state)) :
    Traversal.State<&1,&2,Reader,Accumulator,Context>}:
  +head = Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),Nat.mod(count,8n),state)
  %Projection.head_then_eights(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_fold_step(~Model,~Element,~Accumulator,~Context,~sample,~combine),
    ~model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine),~round_steps(~Model,~Element,~Accumulator,~Context,~sample,~combine),count,state) :
    {Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.div(count,8n),
        Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.mod(count,8n),
          represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state))) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,_) : Traversal.State<&1,&2,Reader,Accumulator,Context>}
  %Equal.sym(Traversal.State<&1,&2,Reader,Accumulator,Context>,
    Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.mod(count,8n),
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,state)),
    represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,head),
    fold_refines_ordered_model(~Reader,~Model,~Element,~Accumulator,~Context,~pack_reader,~read,~sample,~combine,~read_matches,Nat.mod(count,8n),state)) :
    {Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~combine),Nat.div(count,8n),_) ==
      represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader,
        Loop.run(~Traversal.State<&2,&2,Model,Accumulator,Context>,~model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine),Nat.div(count,8n),head)) :
      Traversal.State<&1,&2,Reader,Accumulator,Context>}
  Projection.iteration_refines_model(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.State<&2,&2,Model,Accumulator,Context>,
    ~Traversal.fold_round(~Reader,~Element,~Accumulator,~Context,~read,~combine),~model_round(~Model,~Element,~Accumulator,~Context,~sample,~combine),
    ~represent_fold(~Reader,~Model,~Accumulator,~Context,~pack_reader),
    ~round_refines_model(~Reader,~Model,~Element,~Accumulator,~Context,~pack_reader,~read,~sample,~combine,~read_matches),Nat.div(count,8n),head)

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

# Reading the values projection does not need the reader's representation.
def collect_prepend_values(~Reader: Type,~Element: Data,+value: Element,result: Reader & List<&2,Element>) ->
  {Pair.snd(Reader,List<&2,Element>,Traversal.collect_prepend(~&2,~Reader,~Element,value,result)) == value <> Pair.snd(Reader,List<&2,Element>,result) : List<&2,Element>}:
  (reader,values) = result
  {==}

# One leaf supplies source, value and observation facts. The shared collection
# lifts all three together, including leaf premises for each input key.
def collected_leaf_frame(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,context: Context,key: Key,result: Source & Element) -> Data:
  (+source,+value) = result
  Shape.Both<Frame(context,source),Shape.Both<ValueFrame(context,value),{observe(context,value) == sample(context,key) : Model}>>

def collected_list_frame(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,+context: Context,+keys: List<&2,Key>,result: Source & List<&2,Element>) -> Data:
  (+source,+values) = result
  Shape.Both<Frame(context,source),Shape.Both<Members.All(Element,{value => ValueFrame(context,value) : Element -> Data},values),
    {collect_values_context(~&2,~Model,~Element,~Context,~observe,values,context) ==
      collect_values_context(~&2,~Model,~Key,~Context,~sample,keys,context) : List<&2,Model>}>>

def collect_list_certified_join(~Source: Data,~Element: Data,~Key: Data,~Context: Data,~Model: Data,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Key -> Model,+context: Context,+key: Key,+rest: List<&2,Key>,+value: Element,
  tail: Source & List<&2,Element>,value_frame: ValueFrame(context,value),+value_equal: {observe(context,value) == sample(context,key) : Model},
  tail_frame: collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,rest,tail)) ->
  collected_list_frame(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key <> rest,Traversal.collect_prepend(~&2,~Source,~Element,value,tail)):
  (source,+values) = tail
  Shape.Both{source_frame,Shape.Both{values_frame,+values_equal}} = tail_frame
  Shape.Both{source_frame,Shape.Both{Shape.Both{value_frame,values_frame},
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
  Shape.Both{source_frame,Shape.Both{value_frame,+value_equal}} = head_frame
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
    case Nil{}: Shape.Both{source_frame,Shape.Both{Unit{},{==}}}
    case +key <> +rest:
      Shape.Both{head_frame,+tail_frame} = keys_frame
      collect_list_certified_next(~Source,~Element,~Key,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,key,rest,read(source,project(context),key),
        source => Traversal.collect_list(~&2,~Source,~Element,~Key,~ReadContext,~read,rest,source,project(context)),leaf(source,context,key,source_frame,head_frame),
        source => source_frame => collect_list_certified(~Source,~Element,~Key,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~KeyFrame,~observe,~sample,~read,~leaf,rest,source,context,source_frame,tail_frame))

def collect_values_context_length(~ownership: Quant,~Runtime: Kind(ownership),~Model: Data,~Context: Data,~pack: Context -> Model -> Runtime,
  values: List<&2,Model>,+context: Context) ->
  {List.length(ownership,Runtime,collect_values_context(~ownership,~Runtime,~Model,~Context,~pack,values,context)) == List.length(&2,Model,values) : Nat}:
  match values:
    case Nil{}: {==}
    case head <> +rest:
      Equal.cong(Nat,Nat,value => 1n+value,List.length(ownership,Runtime,collect_values_context(~ownership,~Runtime,~Model,~Context,~pack,rest,context)),
        List.length(&2,Model,rest),collect_values_context_length(~ownership,~Runtime,~Model,~Context,~pack,rest,context))

# A projected context exposes the logical index while the reader carries its
# own counters. The leaf proves the transition used by the common collector.
def collected_indexed_leaf_frame(~Source: Data,~Element: Data,~Context: Data,~Model: Data,~ReadContext: Data,~project: Context -> Nat -> ReadContext,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Nat -> Model,+context: Context,+index: Nat,
  result: Source & Traversal.Collected<&2,ReadContext,Element>) -> Data:
  (+source,Traversal.Collected{next,+value}) = result
  Shape.Both<{next == project(context,1n+index) : ReadContext},
    collected_leaf_frame(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,index,(source,value))>

def collect_certified_next(~Source: Data,~Element: Data,~Context: Data,~Model: Data,~ReadContext: Data,~project: Context -> Nat -> ReadContext,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Nat -> Model,+context: Context,+count: Nat,+index: Nat,
  head: Source & Traversal.Collected<&2,ReadContext,Element>,tail: Source -> ReadContext -> Source & List<&2,Element>,
  head_frame: collected_indexed_leaf_frame(~Source,~Element,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~observe,~sample,context,index,head),
  tail_frame: @+source: Source -> Frame(context,source) ->
    collected_list_frame(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,Ranges.range(count,1n+index),tail(source,project(context,1n+index)))) ->
  collected_list_frame(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,Ranges.range(1n+count,index),
    Traversal.collect_next(~&2,~Source,~Element,~ReadContext,head,tail)):
  (+source,Traversal.Collected{+next,+value}) = head
  Shape.Both{+next_equal,Shape.Both{source_frame,Shape.Both{value_frame,+value_equal}}} = head_frame
  %Equal.sym(ReadContext,next,project(context,1n+index),next_equal) :
    collected_list_frame(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,Ranges.range(1n+count,index),
      Traversal.collect_prepend(~&2,~Source,~Element,value,tail(source,_)))
  collect_list_certified_join(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,index,Ranges.range(count,1n+index),value,
    tail(source,project(context,1n+index)),value_frame,value_equal,tail_frame(source,source_frame))

def collect_certified(~Source: Data,~Element: Data,~Context: Data,~Model: Data,~ReadContext: Data,~project: Context -> Nat -> ReadContext,
  ~Frame: Context -> Source -> Data,~ValueFrame: Context -> Element -> Data,~KeyFrame: Context -> Nat -> Data,
  ~observe: Context -> Element -> Model,~sample: Context -> Nat -> Model,
  ~read: Source -> ReadContext -> Source & Traversal.Collected<&2,ReadContext,Element>,
  ~leaf: @+source: Source -> @+context: Context -> @+index: Nat -> Frame(context,source) -> KeyFrame(context,index) ->
    collected_indexed_leaf_frame(~Source,~Element,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~observe,~sample,context,index,read(source,project(context,index))),
  +count: Nat,+index: Nat,+source: Source,+context: Context,source_frame: Frame(context,source),
  keys_frame: Members.All(Nat,{key => KeyFrame(context,key) : Nat -> Data},Ranges.range(count,index))) ->
  collected_list_frame(~Source,~Element,~Nat,~Context,~Model,~Frame,~ValueFrame,~observe,~sample,context,Ranges.range(count,index),
    Traversal.collect(~&2,~Source,~Element,~ReadContext,~read,count,source,project(context,index))):
  match count:
    case 0n: Shape.Both{source_frame,Shape.Both{Unit{},{==}}}
    case 1n+ +rest:
      Shape.Both{head_frame,+tail_frame} = keys_frame
      collect_certified_next(~Source,~Element,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~observe,~sample,context,rest,index,read(source,project(context,index)),
        source => context => Traversal.collect(~&2,~Source,~Element,~ReadContext,~read,rest,source,context),leaf(source,context,index,source_frame,head_frame),
        source => source_frame => collect_certified(~Source,~Element,~Context,~Model,~ReadContext,~project,~Frame,~ValueFrame,~KeyFrame,~observe,~sample,~read,~leaf,rest,1n+index,source,context,source_frame,tail_frame))

def collect_values_context_range(~Element: Data,~Context: Data,~sample: Context -> Nat -> Element,
  count: Nat,+first: Nat,+context: Context) ->
  {collect_values_context(~&2,~Element,~Nat,~Context,~sample,Ranges.range(count,first),context) ==
    ListRead.enumerate(~Element,~Context,~sample,count,first,context) : List<&2,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Element>,List<&2,Element>,values => sample(context,first) <> values,
        collect_values_context(~&2,~Element,~Nat,~Context,~sample,Ranges.range(rest,1n+first),context),
        ListRead.enumerate(~Element,~Context,~sample,rest,1n+first,context),collect_values_context_range(~Element,~Context,~sample,rest,1n+first,context))

def collect_next_length(~Source: Data,~Element: Data,~Context: Data,+count: Nat,
  head: Source & Traversal.Collected<&2,Context,Element>,tail: Source -> Context -> Source & List<&2,Element>,
  tail_length: @+source: Source -> @+context: Context -> {List.length(&2,Element,Pair.snd(Source,List<&2,Element>,tail(source,context))) == count : Nat}) ->
  {List.length(&2,Element,Pair.snd(Source,List<&2,Element>,Traversal.collect_next(~&2,~Source,~Element,~Context,head,tail))) == 1n+count : Nat}:
  (+source,Traversal.Collected{+context,+value}) = head
  %Equal.sym(List<&2,Element>,Pair.snd(Source,List<&2,Element>,Traversal.collect_prepend(~&2,~Source,~Element,value,tail(source,context))),
    value <> Pair.snd(Source,List<&2,Element>,tail(source,context)),collect_prepend_values(~Source,~Element,value,tail(source,context))) :
    {List.length(&2,Element,_) == 1n+count : Nat}
  Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Element,Pair.snd(Source,List<&2,Element>,tail(source,context))),count,tail_length(source,context))

def collect_length(~Source: Data,~Element: Data,~Context: Data,
  ~read: Source -> Context -> Source & Traversal.Collected<&2,Context,Element>,count: Nat,+source: Source,+context: Context) ->
  {List.length(&2,Element,Pair.snd(Source,List<&2,Element>,Traversal.collect(~&2,~Source,~Element,~Context,~read,count,source,context))) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      collect_next_length(~Source,~Element,~Context,rest,read(source,context),
        source => context => Traversal.collect(~&2,~Source,~Element,~Context,~read,rest,source,context),
        source => context => collect_length(~Source,~Element,~Context,~read,rest,source,context))

# Pointwise equality on the visited keys preserves any collected value list.
def collect_values_context_agrees(~Element: Data,~Key: Data,~LeftContext: Data,~RightContext: Data,
  ~left: LeftContext -> Key -> Element,~right: RightContext -> Key -> Element,
  +keys: List<&2,Key>,+left_context: LeftContext,+right_context: RightContext,
  agreements: Members.All(Key,{key => {left(left_context,key) == right(right_context,key) : Element} : Key -> Data},keys)) ->
  {collect_values_context(~&2,~Element,~Key,~LeftContext,~left,keys,left_context) ==
    collect_values_context(~&2,~Element,~Key,~RightContext,~right,keys,right_context) : List<&2,Element>}:
  match keys:
    case Nil{}: {==}
    case +key <> +rest:
      Shape.Both{head,tail} = agreements
      %Equal.sym(Element,left(left_context,key),right(right_context,key),head) :
        {_ <> collect_values_context(~&2,~Element,~Key,~LeftContext,~left,rest,left_context) ==
          right(right_context,key) <> collect_values_context(~&2,~Element,~Key,~RightContext,~right,rest,right_context) : List<&2,Element>}
      Equal.cong(List<&2,Element>,List<&2,Element>,values => right(right_context,key) <> values,
        collect_values_context(~&2,~Element,~Key,~LeftContext,~left,rest,left_context),
        collect_values_context(~&2,~Element,~Key,~RightContext,~right,rest,right_context),
        collect_values_context_agrees(~Element,~Key,~LeftContext,~RightContext,~left,~right,rest,left_context,right_context,tail))

# A fixed context condition is carried unchanged while its fold is projected.
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
