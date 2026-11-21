# What a list of parts of held source pieces writes. Every part writes the
# cells of its box and leaves the others, so a target cell ends with the value
# of the last part that reaches it, or keeps its own.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_move.bend as Move
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_move_readback.bend as Held
import ./tensor_move_part.bend as Part

# ---- Parts as a model ----

# The parts run in order from a holding into a buffer tree.
def parts_ran(~Element: Data,list: List<&2,Move.Part>,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  match list:
    case Nil{}: Held.Ran{holding,tree}
    case Con{Move.Part{+axes,+source,+target},rest}:
      +ran = Part.part_ran(~Element,axes,source,target,holding,tree)
      parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))

def parts_on_eta(~Element: Data,+rest: List<&2,Move.Part>,pair: Move.Holding<Element> & Storage.Buffer<Element>) ->
  {Move.parts_on(~Element,rest,pair) ==
    Move.parts(~Element,rest,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,pair),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,pair)) : Move.Holding<Element> & Storage.Buffer<Element>}:
  (held,buffer) = pair
  {==}

# The buffer after a part is the model of its tree, as long as before.
def part_content(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))))) == Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)} : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))))),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
    Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)},
    Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
    Held.run_target(~Element,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)),Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model),Part.part_refines(~Element,axes,source,target,holding,model)))

# The runtime parts on a packed holding into a packed buffer are their model.
def parts_refines(~Element: Data,list: List<&2,Move.Part>,+holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,model)),parts_ran(~Element,list,holding,Buffer.tree(~Element,model)),
    Buffer.length(~Element,model)>:
  match list:
    case Nil{}: Held.RunResult{{==},Held.content_parts(~Element,model)}
    case Con{Move.Part{+axes,+source,+target},+rest}:
      +ran = Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))
      +next = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))))
      +content = part_content(~Element,axes,source,target,holding,model)
      +same_tree = Part.model_tree(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      +same_length = Part.model_length(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.parts_on(~Element,rest,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))),
          Move.parts(~Element,rest,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))),parts_on_eta(~Element,rest,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))) :
        Held.RunResult<Element,_,parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Buffer.length(~Element,model)>
      %Equal.sym(Move.Holding<Element>,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model))),Held.pack(~Element,Held.ran_holding(~Element,ran)),
          Held.run_held(~Element,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)),ran,Buffer.length(~Element,model),Part.part_refines(~Element,axes,source,target,holding,model))) :
        Held.RunResult<Element,Move.parts(~Element,rest,_,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))),parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Buffer.length(~Element,model)>
      %Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))) :
        Held.RunResult<Element,Move.parts(~Element,rest,Held.pack(~Element,Held.ran_holding(~Element,ran)),_),
          parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Buffer.length(~Element,model)>
      %same_tree :
        Held.RunResult<Element,Move.parts(~Element,rest,Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next)),
          parts_ran(~Element,rest,Held.ran_holding(~Element,ran),_),Buffer.length(~Element,model)>
      %same_length :
        Held.RunResult<Element,Move.parts(~Element,rest,Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next)),
          parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Buffer.tree(~Element,next)),_>
      parts_refines(~Element,rest,Held.ran_holding(~Element,ran),next)

# ---- What a part keeps ----

def parts_order(~Element: Data,list: List<&2,Move.Part>,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,parts_ran(~Element,list,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match list:
    case Nil{}: {==}
    case Con{Move.Part{+axes,+source,+target},+rest}:
      +ran = Part.part_ran(~Element,axes,source,target,holding,tree)
      Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),
        Held.order(~Element,Held.ran_holding(~Element,ran)),Held.order(~Element,holding),
        parts_order(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Part.part_order(~Element,axes,source,target,holding,tree))

def parts_balanced(~Element: Data,list: List<&2,Move.Part>,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,Held.ran_tree(~Element,parts_ran(~Element,list,holding,tree))):
  match list:
    case Nil{}: balanced
    case Con{Move.Part{+axes,+source,+target},+rest}:
      +ran = Part.part_ran(~Element,axes,source,target,holding,tree)
      parts_balanced(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),depth,Part.part_balanced(~Element,axes,source,target,holding,tree,depth,balanced))

def parts_capacity(~Element: Data,list: List<&2,Move.Part>,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,Held.ran_tree(~Element,parts_ran(~Element,list,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  match list:
    case Nil{}: {==}
    case Con{Move.Part{+axes,+source,+target},+rest}:
      +ran = Part.part_ran(~Element,axes,source,target,holding,tree)
      Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,parts_ran(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),
        Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),
        parts_capacity(~Element,rest,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Part.part_capacity(~Element,axes,source,target,holding,tree))

# ---- The cells parts write ----

# What a part needs to run on a target of capacity cells from holdings of
# source cells in ranges.
def PartReady(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+cap: U32,+word: U32,+ranges: Held.Spans,+capacity: U32) -> Data:
  Held.Both<{Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},
    Held.Both<{Nat.is_lt(0n,Part.total(axes)) == True{} : Bool},
      Held.Both<Part.StridesRoom(axes,word),
        Held.Both<Part.CellsHeld(axes,source,target,ranges,0n,Part.total(axes)),Part.CellsBelow(axes,source,target,capacity,0n,Part.total(axes))>>>>

def PartsReady(list: List<&2,Move.Part>,+cap: U32,+word: U32,+ranges: Held.Spans,+capacity: U32) -> Data:
  match list:
    case Nil{}: Unit
    case Con{Move.Part{+axes,+source,+target},rest}:
      Held.Both<PartReady(axes,source,target,cap,word,ranges,capacity),PartsReady(rest,cap,word,ranges,capacity)>

# A part reaches the target cell query at box index index and at no other.
def HitAt(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+index: Nat,+query: U32) -> Data:
  Held.Both<{Nat.is_lt(index,Part.total(axes)) == True{} : Bool},
    Held.Both<{Part.cell_target(axes,index,source,target) == query : U32},Part.CellsApart(axes,source,target,index,0n,Part.total(axes))>>

type Reached<-Hit: Data,-Miss: Data> is Data:
  Hits{hit: Hit}
  Misses{miss: Miss}

# Part after part, the cell query goes from old to result: a part that
# reaches it at its listed box index writes the value of that source cell, a
# part that misses it leaves it.
def PartsAt(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Move.Part>,indices: List<&2,Nat>,+query: U32,+old: Element,+result: Element) -> Data:
  match list indices:
    case Nil{} indices: {old == result : Element}
    case Con{Move.Part{+axes,+source,+target},rest} Con{+index,more}:
      Reached<Held.Both<HitAt(axes,source,target,index,query),PartsAt(~Element,~Context,~value,context,rest,more,query,value(context,Part.cell_source(axes,index,source,target)),result)>,
        Held.Both<Part.CellsMiss(axes,source,target,query,0n,Part.total(axes)),PartsAt(~Element,~Context,~value,context,rest,more,query,old,result)>>
    case Con{Move.Part{+axes,+source,+target},rest} Nil{}: Empty

# After one part the cell query holds what the later parts start from.
def reached_next(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+index: Nat,+rest: List<&2,Move.Part>,+more: List<&2,Nat>,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},+result: Element,
  +reached: Reached<Held.Both<HitAt(axes,source,target,index,query),PartsAt(~Element,~Context,~value,context,rest,more,query,value(context,Part.cell_source(axes,index,source,target)),result)>,
    Held.Both<Part.CellsMiss(axes,source,target,query,0n,Part.total(axes)),PartsAt(~Element,~Context,~value,context,rest,more,query,Tree.read(Element,tree,query),result)>>,
  +room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,Part.total(axes)) == True{} : Bool},
  +strides: Part.StridesRoom(axes,word),+cells: Part.CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,Part.total(axes)),
  +below: Part.CellsBelow(axes,source,target,Tree.capacity(Element,tree),0n,Part.total(axes))) ->
  PartsAt(~Element,~Context,~value,context,rest,more,query,Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,tree)),query),result):
  match reached:
    case Hits{+hit}:
      Held.Both{Held.Both{+inside,Held.Both{+same,+apart}},+later} = hit
      +after = Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,tree))
      +written = {%same : {Tree.read(Element,after,_) == value(context,Part.cell_source(axes,index,source,target)) : Element}
        Part.part_written(~Element,~Context,~value,context,axes,source,target,holding,tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,room,word,word_ok,strides,
          stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,values,cells,below,index,inside,apart) :
        {Tree.read(Element,after,query) == value(context,Part.cell_source(axes,index,source,target)) : Element}}
      %Equal.sym(Element,Tree.read(Element,after,query),value(context,Part.cell_source(axes,index,source,target)),written) :
        PartsAt(~Element,~Context,~value,context,rest,more,query,_,result)
      later
    case Misses{+miss}:
      Held.Both{+cells_miss,+later} = miss
      +after = Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,tree))
      %Equal.sym(Element,Tree.read(Element,after,query),Tree.read(Element,tree,query),
          Part.part_kept(~Element,axes,source,target,holding,tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,room,positive,word,word_ok,strides,
            stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,cells,below,query,query_inside,cells_miss)) :
        PartsAt(~Element,~Context,~value,context,rest,more,query,_,result)
      later

# Parts that take the cell query from its own value to result leave result
# there.
def parts_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Move.Part>,indices: List<&2,Nat>,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +ready: PartsReady(list,cap,word,Held.spans(~Element,stack),Tree.capacity(Element,tree)),
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},+result: Element,
  +at: PartsAt(~Element,~Context,~value,context,list,indices,query,Tree.read(Element,tree,query),result)) ->
  {Tree.read(Element,Held.ran_tree(~Element,parts_ran(~Element,list,holding,tree)),query) == result : Element}:
  match list indices:
    case Nil{} indices: at
    case Con{Move.Part{+axes,+source,+target},+rest} Con{+index,+more}:
      Held.Both{Held.Both{+room,Held.Both{+positive,Held.Both{+strides,Held.Both{+cells,+below}}}},+later} = ready
      +ran = Part.part_ran(~Element,axes,source,target,holding,tree)
      +same = Equal.sym(U32,Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),Part.part_capacity(~Element,axes,source,target,holding,tree))
      parts_cell(~Element,~Context,~value,context,rest,more,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),depth,
        Part.part_balanced(~Element,axes,source,target,holding,tree,depth,balanced),depth_ok,
        {%same : {U32.is_le(_,2147483648) == True{} : Bool} capacity_ok : {U32.is_le(Tree.capacity(Element,Held.ran_tree(~Element,ran)),2147483648) == True{} : Bool}},
        cap,cap_ok,word,word_ok,stack,
        Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,ran)),Held.order(~Element,holding),stack,Part.part_order(~Element,axes,source,target,holding,tree),ordered),
        start,end,source_capacity,source_ok,bounded,sorted,Part.part_values(~Element,~Context,~value,context,axes,source,target,holding,tree,values),
        {%same : PartsReady(rest,cap,word,Held.spans(~Element,stack),_) later : PartsReady(rest,cap,word,Held.spans(~Element,stack),Tree.capacity(Element,Held.ran_tree(~Element,ran)))},
        query,{%same : {U32.is_lt(query,_) == True{} : Bool} query_inside : {U32.is_lt(query,Tree.capacity(Element,Held.ran_tree(~Element,ran))) == True{} : Bool}},result,
        reached_next(~Element,~Context,~value,context,axes,source,target,index,rest,more,holding,tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,word,word_ok,stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,values,
          query,query_inside,result,at,room,positive,strides,cells,below))
    case Con{Move.Part{+axes,+source,+target},+rest} Nil{}:
      Empty.absurd({Tree.read(Element,Held.ran_tree(~Element,parts_ran(~Element,Con{Move.Part{axes,source,target},rest},holding,tree)),query) == result : Element},at)
