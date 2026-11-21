# A gather, then a map of every target piece it fills: the gather writes a
# piece as its law says, and the map turns what the law holds of the piece
# into what holds after the map.

import Base
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_map.bend as PiecesMap

# The model of a gather, then a map: the holdings the gather leaves, and the
# piece it writes after the map.
def then_model(~Element: Data,~Context: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),~gather_model: Moved.GatherModel(~Element,~Context),
  +context: Tensors.Then<Context,Later>,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Moved.Holdings<Element>) -> Moved.Gathered<Element>:
  Tensors.Then{+own,+later} = context
  +inner = gather_model(own,components,first,count,fill,holdings)
  Moved.Gathered{Moved.gathered_holdings(~Element,inner),
    Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,Moved.gathered_output(~Element,inner))))}

# A context of a gather, then a map, agrees when it maps with later and its
# gather context agrees.
def ThenAgrees(~Element: Data,~Context: Data,~Later: Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  +later: Later,+context: Tensors.Then<Context,Later>,+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element) -> Data:
  match context:
    case Tensors.Then{+own,+other}:
      Held.Both<{other == later : Later},Agrees(own,components,origins,target,target_grid,fill)>

# What a gather gives a piece, the map gives the piece it maps.
def then_facts(~Element: Data,~Context: Data,~Later: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),
  ~after: Tensors.PieceMap(~Element,~Later),~step: @+context: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~rule: PiecesMap.Rule(~Element,~Later,~after,~step),
  ~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Outer: @+later: Later -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~lift: @+later: Later -> @+first: U32 -> @+rows: U32 -> @+inner: Buffer.Content<Element> -> @+holds: Inner(first,rows,inner) -> Outer(later,first,rows,step(later,first,rows,inner)),
  +later: Later,
  +own: Context,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Moved.Holdings<Element>,+models: Moved.Models<Element>,
  +inner_facts: Moved.GatherFacts(~Element,~Context,~gather,~Inner,own,components,first,count,fill,holdings,models,gather_model(own,components,first,count,fill,holdings))) ->
  Moved.GatherFacts(~Element,~Tensors.Then<Context,Later>,~Tensors.then_gather(~Element,~Context,~Later,~gather,~after),~Outer(later),Tensors.Then{own,later},components,first,count,fill,
    holdings,models,then_model(~Element,~Context,~Later,~after,~gather_model,Tensors.Then{own,later},components,first,count,fill,holdings)):
  Held.Both{+same,Held.Both{+order,+target}} = inner_facts
  +inner = gather_model(own,components,first,count,fill,holdings)
  +held = Moved.gathered_holdings(~Element,inner)
  +output = Moved.gathered_output(~Element,inner)
  +stepped = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output)))),Buffer.contents(~Element,after(later,first,count,Buffer.pack(~Element,output))),
    step(later,first,count,Buffer.content(~Element,output)),Buffer.remodel_content(~Element,after(later,first,count,Buffer.pack(~Element,output))),
    Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,after(later,first,count,Buffer.pack(~Element,output))),step(later,first,count,Buffer.contents(~Element,Buffer.pack(~Element,output))),
      step(later,first,count,Buffer.content(~Element,output)),rule(later,first,count,Buffer.pack(~Element,output)),
      Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,content => step(later,first,count,content),Buffer.contents(~Element,Buffer.pack(~Element,output)),
        Buffer.content(~Element,output),Buffer.packed_contents(~Element,output))))
  Held.Both{
    {%Equal.sym(Tensors.Holdings<Element> & Storage.Buffer<Element>,gather(own,components,first,count,fill,Moved.pack_holdings(~Element,holdings)),
          (Moved.pack_holdings(~Element,held),Buffer.pack(~Element,output)),same) :
        {Tensors.then_output(~Element,~Later,~after,later,first,count,_) ==
          (Moved.pack_holdings(~Element,held),Buffer.pack(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output))))) : Tensors.Holdings<Element> & Storage.Buffer<Element>}
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output)))),after(later,first,count,Buffer.pack(~Element,output)),Buffer.remodeled(~Element,after(later,first,count,Buffer.pack(~Element,output)))) :
        {(Moved.pack_holdings(~Element,held),after(later,first,count,Buffer.pack(~Element,output))) == (Moved.pack_holdings(~Element,held),_) : Tensors.Holdings<Element> & Storage.Buffer<Element>}
      {==} :
      {Tensors.then_output(~Element,~Later,~after,later,first,count,gather(own,components,first,count,fill,Moved.pack_holdings(~Element,holdings))) ==
        (Moved.pack_holdings(~Element,held),Buffer.pack(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output))))) : Tensors.Holdings<Element> & Storage.Buffer<Element>}},
    Held.Both{order,
      {%Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output)))),step(later,first,count,Buffer.content(~Element,output)),stepped) :
          Outer(later,first,count,_)
        lift(later,first,count,Buffer.content(~Element,output),target) : Outer(later,first,count,Buffer.content(~Element,Buffer.remodel(~Element,after(later,first,count,Buffer.pack(~Element,output)))))}}}

# The law of a gather, then a map: the law of the gather, with what the map
# keeps of what the gather holds.
def then_law(~Element: Data,~Context: Data,~Later: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: Moved.GatherModel(~Element,~Context),
  ~after: Tensors.PieceMap(~Element,~Later),~step: @+context: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~rule: PiecesMap.Rule(~Element,~Later,~after,~step),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Outer: @+later: Later -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~gathers: Moved.GatherLaw(~Element,~Context,~gather,~gather_model,~Property,~Inner,~Agrees),
  ~lift: @+later: Later -> @+first: U32 -> @+rows: U32 -> @+inner: Buffer.Content<Element> -> @+holds: Inner(first,rows,inner) -> Outer(later,first,rows,step(later,first,rows,inner)),
  +later: Later,
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Tensors.Then<Context,Later>,
  +index: U32,+piece: U32,+models: Moved.Models<Element>,+holdings: Moved.Holdings<Element>,
  +agrees: ThenAgrees(~Element,~Context,~Later,~Agrees,later,context,components,origins,target,target_grid,fill),
  +ready: Moved.TakenReady(~Element,~Property,components,origins,models,index),
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},+order: Moved.HeldOrder(~Element,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(components,target_grid,target,index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Then<Context,Later>,~Tensors.then_gather(~Element,~Context,~Later,~gather,~after),~Outer(later),context,components,
    Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings,models,
    then_model(~Element,~Context,~Later,~after,~gather_model,context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings)):
  match context:
    case Tensors.Then{+own,+other}:
      Held.Both{+same_later,+inner_agrees} = agrees
      +first = Grid.start(target_grid,piece)
      +count = Grid.span(target_grid,piece)
      %Equal.sym(Later,other,later,same_later) :
        Moved.GatherFacts(~Element,~Tensors.Then<Context,Later>,~Tensors.then_gather(~Element,~Context,~Later,~gather,~after),~Outer(later),Tensors.Then{own,_},components,
          first,count,fill,holdings,models,then_model(~Element,~Context,~Later,~after,~gather_model,Tensors.Then{own,_},components,first,count,fill,holdings))
      then_facts(~Element,~Context,~Later,~gather,~gather_model,~after,~step,~rule,~Inner,~Outer,~lift,later,own,components,first,count,fill,holdings,models,
        gathers(components,origins,target,target_grid,fill,own,index,piece,models,holdings,inner_agrees,ready,inside,order,member))

# ---- A gather, then nothing ----

# A piece kept as it is holds what the gather holds of it.
def Kept(~Element: Data,~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+later: Unit,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Inner(first,rows,buffer)

def kept_step(~Element: Data,+context: Unit,+first: U32,+count: U32,content: Buffer.Content<Element>) -> Buffer.Content<Element>:
  content

def kept_rule(~Element: Data,+context: Unit,+first: U32,+count: U32,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Tensors.unchanged(~Element,context,first,count,buffer)) == kept_step(~Element,context,first,count,Buffer.contents(~Element,buffer)) : Buffer.Content<Element>}:
  {==}

def kept_lift(~Element: Data,~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+later: Unit,+first: U32,+rows: U32,+inner: Buffer.Content<Element>,
  +holds: Inner(first,rows,inner)) -> Kept(~Element,~Inner,later,first,rows,kept_step(~Element,later,first,rows,inner)):
  holds

# A piece kept as the gather wrote it holds what the gather holds of it.
def kept_inner(~Element: Data,~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  held: Kept(~Element,~Inner,Unit{},first,rows,buffer)) -> Inner(first,rows,buffer):
  held
