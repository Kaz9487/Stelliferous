# Readback through a projection of stored elements preserves addresses and order.
import Base
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program_every.bend as Every
import ./boolean_logic.bend as Boolean
import ../kernel_gemm.bend as KernelGemm

def projected(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,writes: List<&2,Writes.Assignment<Element>>,+context: Context) -> List<&2,Writes.Assignment<Value>>:
  match writes:
    case Nil{}: []
    case Writes.Put{index,value} <> rest: Writes.Put{index,observe(context,value)} <> projected(~Element,~Value,~Context,~observe,rest,context)

type Query<-Context: Data,-Value: Data> is Data:
  Query{context: Context,address: U32,wanted: Value}

def Holds(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+query: Query<Context,Value>,address: U32,value: Element) -> Data:
  Query{context,at,wanted} = query
  Boolean.When(U32.is_eq(address,at),{observe(context,value) == wanted : Value})

def matching(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,+query: U32,+wanted: Value,
  writes: List<&2,Writes.Assignment<Element>>,
  holds: Every.Every(~Element,~Query<Context,Value>,~Holds(~Element,~Value,~Context,~observe),Query{context,query,wanted},writes)) ->
  Readback.Matching(~Value,query,wanted,projected(~Element,~Value,~Context,~observe,writes,context)):
  match writes:
    case Nil{}: Unit{}
    case Writes.Put{address,value} <> rest:
      KernelGemm.Both{here,later} = holds
      KernelGemm.Both{here,matching(~Element,~Value,~Context,~observe,context,query,wanted,rest,later)}

def picked(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,choose: Bool,+left: Element,+right: Element) ->
  {observe(context,Bool.pick(Element,choose,left,right)) == Bool.pick(Value,choose,observe(context,left),observe(context,right)) : Value}:
  match choose:
    case True{}: {==}
    case False{}: {==}

def selected(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,writes: List<&2,Writes.Assignment<Element>>,+query: U32,+previous: Element) ->
  {observe(context,Readback.selected(~Element,writes,query,previous)) == Readback.selected(~Value,projected(~Element,~Value,~Context,~observe,writes,context),query,observe(context,previous)) : Value}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      %picked(~Element,~Value,~Context,~observe,context,U32.is_eq(index,query),value,previous) :
        {observe(context,Readback.selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))) ==
          Readback.selected(~Value,projected(~Element,~Value,~Context,~observe,rest,context),query,_) : Value}
      selected(~Element,~Value,~Context,~observe,context,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))

def contains(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,writes: List<&2,Writes.Assignment<Element>>,+query: U32) ->
  {Readback.contains(~Value,query,projected(~Element,~Value,~Context,~observe,writes,context)) == Readback.contains(~Element,query,writes) : Bool}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,value} <> +rest:
      Equal.cong(Bool,Bool,tail => U32.is_eq(index,query) || tail,Readback.contains(~Value,query,projected(~Element,~Value,~Context,~observe,rest,context)),Readback.contains(~Element,query,rest),
        contains(~Element,~Value,~Context,~observe,context,rest,query))

def appended(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>) ->
  {projected(~Element,~Value,~Context,~observe,List.append(&2,Writes.Assignment<Element>,left,right),context) ==
    List.append(&2,Writes.Assignment<Value>,projected(~Element,~Value,~Context,~observe,left,context),projected(~Element,~Value,~Context,~observe,right,context)) : List<&2,Writes.Assignment<Value>>}:
  match left:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Equal.cong(List<&2,Writes.Assignment<Value>>,List<&2,Writes.Assignment<Value>>,tail => Writes.Put{index,observe(context,value)} <> tail,
        projected(~Element,~Value,~Context,~observe,List.append(&2,Writes.Assignment<Element>,rest,right),context),
        List.append(&2,Writes.Assignment<Value>,projected(~Element,~Value,~Context,~observe,rest,context),projected(~Element,~Value,~Context,~observe,right,context)),
        appended(~Element,~Value,~Context,~observe,context,rest,right))

def covered(~Element: Data,~Value: Data,~Context: Data,~observe: Context -> Element -> Value,+context: Context,+writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,+wanted: Value,
  +balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  +inside: Readback.Inside(~Element,Tree.capacity(Element,tree),writes),
  +matching: Readback.Matching(~Value,query,wanted,projected(~Element,~Value,~Context,~observe,writes,context)),+covered: {Readback.contains(~Element,query,writes) == True{} : Bool}) ->
  {observe(context,Tree.read(Element,Writes.pure(Element,writes,tree),query)) == wanted : Value}:
  +previous = Tree.read(Element,tree,query)
  +observed = projected(~Element,~Value,~Context,~observe,writes,context)
  %Equal.sym(Element,Tree.read(Element,Writes.pure(Element,writes,tree),query),Readback.selected(~Element,writes,query,previous),Readback.program(~Element,writes,depth,tree,query,balanced,depth_ok,query_inside,inside)) :
    {observe(context,_) == wanted : Value}
  Equal.trans(Value,observe(context,Readback.selected(~Element,writes,query,previous)),Readback.selected(~Value,observed,query,observe(context,previous)),wanted,
    selected(~Element,~Value,~Context,~observe,context,writes,query,previous),
    %Equal.trans(Bool,Readback.contains(~Value,query,observed),Readback.contains(~Element,query,writes),True{},
      contains(~Element,~Value,~Context,~observe,context,writes,query),covered) :
      {Readback.selected(~Value,observed,query,observe(context,previous)) == Bool.pick(Value,_,wanted,observe(context,previous)) : Value}
    Readback.selected_matching(~Value,observed,query,wanted,observe(context,previous),matching))
