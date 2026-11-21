# Tensors.moved_with: every component takes its source pieces out, gathers its
# target pieces from them, and the target pieces are placed in the grid.
import Base
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_move.bend as Move
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_move_readback.bend as Held
import ../kernel_gemm.bend as KernelGemm
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_walk.bend as Walk
import ./traversal_partition_refinement.bend as PartitionLaws

# ---- Taking the source pieces of a component ----

# The empty buffer a taken piece leaves behind.
def empty(~Element: Data,+fill: Element) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))

def taken_leaf(~Element: Data,+piece: U32,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) -> PiecesModel.Model<Element>:
  +first = Grid.start(grid,piece)
  PiecesModel.Leaf{Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill)),first,Grid.span(grid,piece)}

def taken_rest(~Element: Data,+piece: U32,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) -> PiecesModel.Model<Element>:
  +first = Grid.start(grid,piece)
  Placed.put(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill))

# The source pieces left after the listed pieces are taken out in order, and
# the pieces taken, as Tensors.taken_inputs takes them.
type Taken<-Element: Data> is Data:
  Taken{rest: PiecesModel.Model<Element>,taken: PiecesModel.Model<Element>}

def joined(~Element: Data,+leaf: PiecesModel.Model<Element>,+after: Taken<Element>) -> Taken<Element>:
  Taken{+rest,+taken} = after
  Taken{rest,PiecesModel.Fork{leaf,taken}}

# The piece of first cell first and count cells already taken out, then the
# listed pieces taken out of the rest in order, as Tensors.taken_on takes them.
def taken_on_model(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+rest: PiecesModel.Model<Element>,
  +data: Buffer.Model<Element>,+first: U32,+count: U32) -> Taken<Element>:
  match pieces:
    case Nil{}:
      Taken{rest,PiecesModel.Leaf{data,first,count}}
    case Con{+piece,+more}:
      +next = Grid.start(grid,piece)
      joined(~Element,PiecesModel.Leaf{data,first,count},taken_on_model(~Element,more,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,rest),Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)),next,Grid.span(grid,piece)))

def taken_model(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) -> Taken<Element>:
  match pieces:
    case Nil{}:
      Taken{model,PiecesModel.Leaf{empty(~Element,fill),0,0}}
    case Con{+piece,+rest}:
      +first = Grid.start(grid,piece)
      taken_on_model(~Element,rest,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,model),Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill)),first,Grid.span(grid,piece))

def packed_pair(~Element: Data,+after: Taken<Element>) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  Taken{+rest,+taken} = after
  (PiecesModel.pack(~Element,rest),PiecesModel.pack(~Element,taken))

# Exchanging an empty buffer for the piece at cell takes its model out.
def exchange_empty(~Element: Data,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+cell: U32,+fill: Element) ->
  {Tensors.exchange(~Element,PiecesModel.pack(~Element,model),layout,cell,Tensors.buffer(~Element,fill,0)) ==
    (PiecesModel.pack(~Element,Placed.put(~Element,model,layout,Tensors.side_of(layout,cell),cell,empty(~Element,fill))),
      Buffer.pack(~Element,Placed.taken(~Element,model,layout,Tensors.side_of(layout,cell),cell,empty(~Element,fill)))) : Pieces.Pieces(~Element) & Storage.Buffer<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,empty(~Element,fill)),Tensors.buffer(~Element,fill,0),Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,0))) :
    {Tensors.exchange(~Element,PiecesModel.pack(~Element,model),layout,cell,_) ==
      (PiecesModel.pack(~Element,Placed.put(~Element,model,layout,Tensors.side_of(layout,cell),cell,empty(~Element,fill))),
        Buffer.pack(~Element,Placed.taken(~Element,model,layout,Tensors.side_of(layout,cell),cell,empty(~Element,fill)))) : Pieces.Pieces(~Element) & Storage.Buffer<Element>}
  Placed.exchanged_put(~Element,model,layout,Tensors.side_of(layout,cell),cell,empty(~Element,fill))

# Joining a taken piece to a model pair and packing it agree.
def taken_joined_pair(~Element: Data,+data: Buffer.Model<Element>,+first: U32,+count: U32,+after: Taken<Element>) ->
  {Tensors.taken_joined(~Element,Buffer.pack(~Element,data),first,count,packed_pair(~Element,after)) ==
    packed_pair(~Element,joined(~Element,PiecesModel.Leaf{data,first,count},after)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}:
  Taken{+rest,+taken} = after
  {==}

def taken_on_exact(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+rest: PiecesModel.Model<Element>,
  +data: Buffer.Model<Element>,+first: U32,+count: U32) ->
  {Tensors.taken_on(~Element,pieces,(PiecesModel.pack(~Element,rest),Buffer.pack(~Element,data)),grid,layout,fill,first,count) ==
    packed_pair(~Element,taken_on_model(~Element,pieces,grid,layout,fill,rest,data,first,count)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}:
  match pieces:
    case Nil{}:
      {==}
    case Con{+piece,+more}:
      +next = Grid.start(grid,piece)
      %Equal.sym(Pieces.Pieces(~Element) & Storage.Buffer<Element>,Tensors.exchange(~Element,PiecesModel.pack(~Element,rest),layout,next,Tensors.buffer(~Element,fill,0)),
          (PiecesModel.pack(~Element,taken_rest(~Element,piece,grid,layout,fill,rest)),Buffer.pack(~Element,Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)))),exchange_empty(~Element,rest,layout,next,fill)) :
        {Tensors.taken_joined(~Element,Buffer.pack(~Element,data),first,count,Tensors.taken_on(~Element,more,_,grid,layout,fill,next,Grid.span(grid,piece))) ==
          packed_pair(~Element,taken_on_model(~Element,Con{piece,more},grid,layout,fill,rest,data,first,count)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
      %Equal.sym(Pieces.Pieces(~Element) & Pieces.Pieces(~Element),Tensors.taken_on(~Element,more,(PiecesModel.pack(~Element,taken_rest(~Element,piece,grid,layout,fill,rest)),Buffer.pack(~Element,Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)))),grid,layout,fill,next,Grid.span(grid,piece)),
          packed_pair(~Element,taken_on_model(~Element,more,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,rest),Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)),next,Grid.span(grid,piece))),
          taken_on_exact(~Element,more,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,rest),Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)),next,Grid.span(grid,piece))) :
        {Tensors.taken_joined(~Element,Buffer.pack(~Element,data),first,count,_) ==
          packed_pair(~Element,taken_on_model(~Element,Con{piece,more},grid,layout,fill,rest,data,first,count)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
      taken_joined_pair(~Element,data,first,count,taken_on_model(~Element,more,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,rest),Placed.taken(~Element,rest,layout,Tensors.side_of(layout,next),next,empty(~Element,fill)),next,Grid.span(grid,piece)))

def taken_exact(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) ->
  {Tensors.taken_inputs(~Element,pieces,grid,layout,fill,PiecesModel.pack(~Element,model)) ==
    packed_pair(~Element,taken_model(~Element,pieces,grid,layout,fill,model)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}:
  match pieces:
    case Nil{}:
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,empty(~Element,fill)),Tensors.buffer(~Element,fill,0),Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,0))) :
        {(PiecesModel.pack(~Element,model),Partition.Leaf{_,0,0,Unit{}}) ==
          (PiecesModel.pack(~Element,model),PiecesModel.pack(~Element,PiecesModel.Leaf{empty(~Element,fill),0,0})) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
      {==}
    case Con{+piece,+rest}:
      +first = Grid.start(grid,piece)
      %Equal.sym(Pieces.Pieces(~Element) & Storage.Buffer<Element>,Tensors.exchange(~Element,PiecesModel.pack(~Element,model),layout,first,Tensors.buffer(~Element,fill,0)),
          (PiecesModel.pack(~Element,taken_rest(~Element,piece,grid,layout,fill,model)),Buffer.pack(~Element,Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill)))),exchange_empty(~Element,model,layout,first,fill)) :
        {Tensors.taken_on(~Element,rest,_,grid,layout,fill,first,Grid.span(grid,piece)) ==
          packed_pair(~Element,taken_model(~Element,Con{piece,rest},grid,layout,fill,model)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
      taken_on_exact(~Element,rest,grid,layout,fill,taken_rest(~Element,piece,grid,layout,fill,model),Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill)),first,Grid.span(grid,piece))

# The spans taken before and the listed pieces, the last taken first.
def taken_gone(pieces: List<&2,U32>,+grid: Grid.Grid,+gone: Held.Spans) -> Held.Spans:
  match pieces:
    case Nil{}:
      gone
    case Con{+piece,rest}:
      taken_gone(rest,grid,Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),gone})

# Every listed piece starts a leaf of the layout with its cells, and no span
# taken before it starts there.
def TakeFacts(pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+gone: Held.Spans) -> Data:
  match pieces:
    case Nil{}:
      Unit
    case Con{+piece,rest}:
      Held.Both<Placed.LeafAt(layout,Grid.start(grid,piece)),Held.Both<{Placed.rows_of(layout,Grid.start(grid,piece)) == Grid.span(grid,piece) : U32},
        Held.Both<Placed.Fresh(gone,Grid.start(grid,piece)),TakeFacts(rest,grid,layout,Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),gone})>>>

# The source left is good up to the spans taken, and every taken piece holds
# the property.
def TakenFacts(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,after: Taken<Element>,+gone: Held.Spans) -> Data:
  match after:
    case Taken{+rest,+taken}:
      Held.Both<Placed.Good(~Element,~Property,rest,gone),PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,taken))>

def joined_facts(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+leaf: PiecesModel.Model<Element>,+after: Taken<Element>,
  +gone: Held.Spans,+holds: PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,leaf)),+facts: TakenFacts(~Element,~Property,after,gone)) ->
  TakenFacts(~Element,~Property,joined(~Element,leaf,after),gone):
  match after:
    case Taken{+rest,+taken}:
      Held.Both{+good,+all} = facts
      Held.Both{good,KernelGemm.Both{holds,all}}

def taken_facts(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,pieces: List<&2,U32>,+grid: Grid.Grid,
  +layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>,+gone: Held.Spans,+same: {PiecesModel.layout(~Element,model) == layout : Pieces.Ranges},
  +ordered: Placed.Ordered(layout),+nothing: Property(0,0,Buffer.content(~Element,empty(~Element,fill))),+facts: TakeFacts(pieces,grid,layout,gone),
  +good: Placed.Good(~Element,~Property,model,gone)) ->
  TakenFacts(~Element,~Property,taken_model(~Element,pieces,grid,layout,fill,model),taken_gone(pieces,grid,gone)):
  match pieces:
    case Nil{}:
      Held.Both{good,nothing}
    case Con{+piece,Nil{}}:
      Held.Both{+at,+rest_facts} = facts
      Held.Both{+rows,+fresh_facts} = rest_facts
      Held.Both{+fresh,+later} = fresh_facts
      +first = Grid.start(grid,piece)
      +back = Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,model),layout,same)
      Held.Both{
        {%same : Placed.Good(~Element,~Property,Placed.put(~Element,model,_,Tensors.side_of(_,first),first,empty(~Element,fill)),Held.Spanned{first,Grid.span(grid,piece),gone})
          Placed.put_gone(~Element,~Property,model,first,Grid.span(grid,piece),empty(~Element,fill),gone,
            {%back : Placed.Ordered(_) ordered : Placed.Ordered(PiecesModel.layout(~Element,model))},
            {%back : Placed.LeafAt(_,first) at : Placed.LeafAt(PiecesModel.layout(~Element,model),first)},good) :
          Placed.Good(~Element,~Property,taken_rest(~Element,piece,grid,layout,fill,model),Held.Spanned{first,Grid.span(grid,piece),gone})},
        {%rows : Property(first,_,Buffer.content(~Element,Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill))))
          %same : Property(first,Placed.rows_of(_,first),Buffer.content(~Element,Placed.taken(~Element,model,_,Tensors.side_of(_,first),first,empty(~Element,fill))))
          Placed.taken_holds(~Element,~Property,model,first,empty(~Element,fill),gone,
            {%back : Placed.Ordered(_) ordered : Placed.Ordered(PiecesModel.layout(~Element,model))},
            {%back : Placed.LeafAt(_,first) at : Placed.LeafAt(PiecesModel.layout(~Element,model),first)},fresh,good) :
          PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,taken_leaf(~Element,piece,grid,layout,fill,model)))}}
    case Con{+piece,Con{+next,+more}}:
      Held.Both{+at,+rest_facts} = facts
      Held.Both{+rows,+fresh_facts} = rest_facts
      Held.Both{+fresh,+later} = fresh_facts
      +first = Grid.start(grid,piece)
      +span = Grid.span(grid,piece)
      +back = Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,model),layout,same)
      +left = taken_rest(~Element,piece,grid,layout,fill,model)
      +model_ordered = {%back : Placed.Ordered(_) ordered : Placed.Ordered(PiecesModel.layout(~Element,model))}
      +model_at = {%back : Placed.LeafAt(_,first) at : Placed.LeafAt(PiecesModel.layout(~Element,model),first)}
      joined_facts(~Element,~Property,taken_leaf(~Element,piece,grid,layout,fill,model),taken_model(~Element,Con{next,more},grid,layout,fill,left),
        taken_gone(Con{next,more},grid,Held.Spanned{first,span,gone}),
        {%rows : Property(first,_,Buffer.content(~Element,Placed.taken(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill))))
          %same : Property(first,Placed.rows_of(_,first),Buffer.content(~Element,Placed.taken(~Element,model,_,Tensors.side_of(_,first),first,empty(~Element,fill))))
          Placed.taken_holds(~Element,~Property,model,first,empty(~Element,fill),gone,model_ordered,model_at,fresh,good) :
          PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,taken_leaf(~Element,piece,grid,layout,fill,model)))},
        taken_facts(~Element,~Property,Con{next,more},grid,layout,fill,left,Held.Spanned{first,span,gone},
          Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,model),layout,
            Placed.put_layout(~Element,model,layout,Tensors.side_of(layout,first),first,empty(~Element,fill)),same),
          ordered,nothing,later,
          {%same : Placed.Good(~Element,~Property,Placed.put(~Element,model,_,Tensors.side_of(_,first),first,empty(~Element,fill)),Held.Spanned{first,span,gone})
            Placed.put_gone(~Element,~Property,model,first,span,empty(~Element,fill),gone,model_ordered,model_at,good) :
            Placed.Good(~Element,~Property,left,Held.Spanned{first,span,gone})}))

# ---- Taking the source pieces of every source ----

type Models<-Element: Data> is Data:
  NoModels{}
  Models{model: PiecesModel.Model<Element>,rest: Models<Element>}

def pack_sources(~Element: Data,models: Models<Element>) -> Tensors.Sources<Element>:
  match models:
    case NoModels{}:
      Tensors.Ended{}
    case Models{model,rest}:
      Tensors.Source{PiecesModel.pack(~Element,model),pack_sources(~Element,rest)}

# The sources left and the pieces taken of every source.
type Took<-Element: Data> is Data:
  Took{left: Models<Element>,taken: Models<Element>}

def took_pair(~Element: Data,+took: Took<Element>) -> Tensors.Sources<Element> & Tensors.Sources<Element>:
  Took{+left,+taken} = took
  (pack_sources(~Element,left),pack_sources(~Element,taken))

def took_joined(~Element: Data,+after: Taken<Element>,+later: Took<Element>) -> Took<Element>:
  Taken{+rest,+taken} = after
  Took{+left,+taken_rest} = later
  Took{Models{rest,left},Models{taken,taken_rest}}

def source_pieces(+components: Tasks.Components,+source: U32,+index: U32,+origin: Tensors.Origin) -> List<&2,U32>:
  Tensors.Origin{+shape,+grid,+ranges,+offset} = origin
  Tasks.box_pieces(grid,shape,Tasks.source_box(components,source,grid,shape,offset,index))

def origin_grid(+origin: Tensors.Origin) -> Grid.Grid:
  Tensors.Origin{+shape,+grid,+ranges,+offset} = origin
  grid

def origin_ranges(+origin: Tensors.Origin) -> Pieces.Ranges:
  Tensors.Origin{+shape,+grid,+ranges,+offset} = origin
  ranges

# The pieces of component index taken out of every source, as
# Tensors.taken_sources takes them.
def took_model(~Element: Data,origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>) -> Took<Element>:
  match origins models:
    case Con{+origin,origins} Models{+model,+rest}:
      took_joined(~Element,taken_model(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model),
        took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest))
    case origins models:
      Took{models,NoModels{}}

def took_bound(~Element: Data,+after: Taken<Element>,+later: Took<Element>) ->
  {Tensors.sources_joined(~Element,packed_pair(~Element,after),took_pair(~Element,later)) ==
    took_pair(~Element,took_joined(~Element,after,later)) : Tensors.Sources<Element> & Tensors.Sources<Element>}:
  Taken{+rest,+taken} = after
  Took{+left,+taken_rest} = later
  {==}

def took_exact(~Element: Data,origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>) ->
  {Tensors.taken_sources(~Element,origins,components,source,index,fill,pack_sources(~Element,models)) ==
    took_pair(~Element,took_model(~Element,origins,components,source,index,fill,models)) : Tensors.Sources<Element> & Tensors.Sources<Element>}:
  match origins models:
    case Nil{} +models:
      {==}
    case Con{Tensors.Origin{+shape,+grid,+ranges,+offset},+origins} Models{+model,+rest}:
      +pieces = Tasks.box_pieces(grid,shape,Tasks.source_box(components,source,grid,shape,offset,index))
      +after = taken_model(~Element,pieces,grid,ranges,fill,model)
      +later = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Pieces.Pieces(~Element) & Pieces.Pieces(~Element),Tensors.taken_inputs(~Element,pieces,grid,ranges,fill,PiecesModel.pack(~Element,model)),packed_pair(~Element,after),
          taken_exact(~Element,pieces,grid,ranges,fill,model)) :
        {Tensors.sources_joined(~Element,_,Tensors.taken_sources(~Element,origins,components,(source + 1 : U32),index,fill,pack_sources(~Element,rest))) ==
          took_pair(~Element,took_joined(~Element,after,later)) : Tensors.Sources<Element> & Tensors.Sources<Element>}
      %Equal.sym(Tensors.Sources<Element> & Tensors.Sources<Element>,Tensors.taken_sources(~Element,origins,components,(source + 1 : U32),index,fill,pack_sources(~Element,rest)),
          took_pair(~Element,later),took_exact(~Element,origins,components,(source + 1 : U32),index,fill,rest)) :
        {Tensors.sources_joined(~Element,packed_pair(~Element,after),_) ==
          took_pair(~Element,took_joined(~Element,after,later)) : Tensors.Sources<Element> & Tensors.Sources<Element>}
      took_bound(~Element,after,later)
    case Con{+origin,+origins} NoModels{}:
      {==}

# The spans taken from every source so far.
type Gones is Data:
  NoGones{}
  Gones{gone: Held.Spans,rest: Gones}

def left_of(~Element: Data,+after: Taken<Element>) -> PiecesModel.Model<Element>:
  Taken{+rest,+taken} = after
  rest

def joined_left(~Element: Data,+leaf: PiecesModel.Model<Element>,+after: Taken<Element>) ->
  {left_of(~Element,joined(~Element,leaf,after)) == left_of(~Element,after) : PiecesModel.Model<Element>}:
  Taken{+rest,+taken} = after
  {==}

# Taking pieces out keeps the layout of the source.
def taken_layout(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) ->
  {PiecesModel.layout(~Element,left_of(~Element,taken_model(~Element,pieces,grid,layout,fill,model))) == PiecesModel.layout(~Element,model) : Pieces.Ranges}:
  match pieces:
    case Nil{}:
      {==}
    case Con{+piece,Nil{}}:
      Placed.put_layout(~Element,model,layout,Tensors.side_of(layout,Grid.start(grid,piece)),Grid.start(grid,piece),empty(~Element,fill))
    case Con{+piece,Con{+next,+more}}:
      +left = taken_rest(~Element,piece,grid,layout,fill,model)
      +after = taken_model(~Element,Con{next,more},grid,layout,fill,left)
      Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,left_of(~Element,joined(~Element,taken_leaf(~Element,piece,grid,layout,fill,model),after))),
        PiecesModel.layout(~Element,left_of(~Element,after)),PiecesModel.layout(~Element,model),
        Equal.cong(PiecesModel.Model<Element>,Pieces.Ranges,PiecesModel.layout(~Element),left_of(~Element,joined(~Element,taken_leaf(~Element,piece,grid,layout,fill,model),after)),
          left_of(~Element,after),joined_left(~Element,taken_leaf(~Element,piece,grid,layout,fill,model),after)),
        Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,left_of(~Element,after)),PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,model),
          taken_layout(~Element,Con{next,more},grid,layout,fill,left),
          Placed.put_layout(~Element,model,layout,Tensors.side_of(layout,Grid.start(grid,piece)),Grid.start(grid,piece),empty(~Element,fill))))

# ---- The inputs of every component ----

def took_left(~Element: Data,+took: Took<Element>) -> Models<Element>:
  Took{+left,+taken} = took
  left

def took_taken(~Element: Data,+took: Took<Element>) -> Models<Element>:
  Took{+left,+taken} = took
  taken

def took_split(~Element: Data,+took: Took<Element>) ->
  {took_pair(~Element,took) == (pack_sources(~Element,took_left(~Element,took)),pack_sources(~Element,took_taken(~Element,took))) : Tensors.Sources<Element> & Tensors.Sources<Element>}:
  Took{+left,+taken} = took
  {==}

# The sources left after the components of a layout took their pieces in
# order, and the pieces every component took.
type Built<-Element: Data> is Data:
  Built{left: Models<Element>,inputs: PiecesModel.Shaped<Models<Element>>}

def built_left(~Element: Data,+built: Built<Element>) -> Models<Element>:
  Built{+left,+inputs} = built
  left

def built_inputs(~Element: Data,+built: Built<Element>) -> PiecesModel.Shaped<Models<Element>>:
  Built{+left,+inputs} = built
  inputs

def built_model(~Element: Data,layout: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+fill: Element,+models: Models<Element>) -> Built<Element>:
  match layout:
    case Pieces.Range{+first,+count}:
      +took = took_model(~Element,origins,components,0,first,fill,models)
      Built{took_left(~Element,took),PiecesModel.Piece{took_taken(~Element,took),first,count}}
    case Pieces.Split{+left,+right}:
      +before = built_model(~Element,left,origins,components,fill,models)
      +after = built_model(~Element,right,origins,components,fill,built_left(~Element,before))
      Built{built_left(~Element,after),PiecesModel.Pieces{built_inputs(~Element,before),built_inputs(~Element,after)}}

def built_pair(~Element: Data,+built: Built<Element>) -> Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>:
  (pack_sources(~Element,built_left(~Element,built)),PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),built_inputs(~Element,built)))

def built_exact(~Element: Data,~Context: Data,layout: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+target: List<&2,U32>,
  +target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>) ->
  {Pieces.build(~Tensors.Sources<Element>,~Tensors.Sources<Element>,~Tensors.Moving<Element,Context>,~Tensors.component_inputs(~Element,~Context),layout,
      Tensors.Moving{components,origins,target,target_grid,fill,context},pack_sources(~Element,models)) ==
    built_pair(~Element,built_model(~Element,layout,origins,components,fill,models)) : Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>}:
  match layout:
    case Pieces.Range{+first,+count}:
      +took = took_model(~Element,origins,components,0,first,fill,models)
      %Equal.sym(Tensors.Sources<Element> & Tensors.Sources<Element>,Tensors.taken_sources(~Element,origins,components,0,first,fill,pack_sources(~Element,models)),
          took_pair(~Element,took),took_exact(~Element,origins,components,0,first,fill,models)) :
        {Tensors.inputs_leaf(~Element,first,count,_) ==
          built_pair(~Element,built_model(~Element,Pieces.Range{first,count},origins,components,fill,models)) : Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>}
      %Equal.sym(Tensors.Sources<Element> & Tensors.Sources<Element>,took_pair(~Element,took),
          (pack_sources(~Element,took_left(~Element,took)),pack_sources(~Element,took_taken(~Element,took))),took_split(~Element,took)) :
        {Tensors.inputs_leaf(~Element,first,count,_) ==
          built_pair(~Element,built_model(~Element,Pieces.Range{first,count},origins,components,fill,models)) : Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>}
      {==}
    case Pieces.Split{+left,+right}:
      +before = built_model(~Element,left,origins,components,fill,models)
      +context_value = {Tensors.Moving{components,origins,target,target_grid,fill,context} : Tensors.Moving<Element,Context>}
      %Equal.sym(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>,
          Pieces.build(~Tensors.Sources<Element>,~Tensors.Sources<Element>,~Tensors.Moving<Element,Context>,~Tensors.component_inputs(~Element,~Context),left,context_value,pack_sources(~Element,models)),
          built_pair(~Element,before),built_exact(~Element,~Context,left,origins,components,target,target_grid,fill,context,models)) :
        {Storage.bind_pair(~Tensors.Sources<Element>,~Partition.Tree<&1,Tensors.Sources<Element>,Unit>,~(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>),_,
            state => left_tree => Storage.bind_pair(~Tensors.Sources<Element>,~Partition.Tree<&1,Tensors.Sources<Element>,Unit>,~(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>),
              Pieces.build(~Tensors.Sources<Element>,~Tensors.Sources<Element>,~Tensors.Moving<Element,Context>,~Tensors.component_inputs(~Element,~Context),right,context_value,state),
              state => right_tree => (state,Partition.Fork{left_tree,right_tree}))) ==
          built_pair(~Element,built_model(~Element,Pieces.Split{left,right},origins,components,fill,models)) : Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>}
      %Equal.sym(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>,
          Pieces.build(~Tensors.Sources<Element>,~Tensors.Sources<Element>,~Tensors.Moving<Element,Context>,~Tensors.component_inputs(~Element,~Context),right,context_value,
            pack_sources(~Element,built_left(~Element,before))),
          built_pair(~Element,built_model(~Element,right,origins,components,fill,built_left(~Element,before))),
          built_exact(~Element,~Context,right,origins,components,target,target_grid,fill,context,built_left(~Element,before))) :
        {Storage.bind_pair(~Tensors.Sources<Element>,~Partition.Tree<&1,Tensors.Sources<Element>,Unit>,~(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>),_,
            state => right_tree => (state,Partition.Fork{PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),built_inputs(~Element,before)),right_tree})) ==
          built_pair(~Element,built_model(~Element,Pieces.Split{left,right},origins,components,fill,models)) : Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>}
      {==}

# ---- From one component to the next ----

def taken_of(~Element: Data,+after: Taken<Element>) -> PiecesModel.Model<Element>:
  Taken{+rest,+taken} = after
  taken

def taken_good(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+after: Taken<Element>,+gone: Held.Spans,
  +facts: TakenFacts(~Element,~Property,after,gone)) -> Placed.Good(~Element,~Property,left_of(~Element,after),gone):
  match after:
    case Taken{+rest,+taken}:
      Held.Both{+good,+all} = facts
      good

def taken_all(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+after: Taken<Element>,+gone: Held.Spans,
  +facts: TakenFacts(~Element,~Property,after,gone)) -> PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,taken_of(~Element,after))):
  match after:
    case Taken{+rest,+taken}:
      Held.Both{+good,+all} = facts
      all

def joined_left_models(~Element: Data,+after: Taken<Element>,+later: Took<Element>) ->
  {took_left(~Element,took_joined(~Element,after,later)) == Models{left_of(~Element,after),took_left(~Element,later)} : Models<Element>}:
  Taken{+rest,+taken} = after
  Took{+left,+taken_rest} = later
  {==}

def joined_taken_models(~Element: Data,+after: Taken<Element>,+later: Took<Element>) ->
  {took_taken(~Element,took_joined(~Element,after,later)) == Models{taken_of(~Element,after),took_taken(~Element,later)} : Models<Element>}:
  Taken{+rest,+taken} = after
  Took{+left,+taken_rest} = later
  {==}

# The spans of every source after component index takes its pieces.
def gones_after(origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,gones: Gones) -> Gones:
  match origins gones:
    case Con{+origin,origins} Gones{+gone,gones}:
      Gones{taken_gone(source_pieces(components,source,index,origin),origin_grid(origin),gone),gones_after(origins,components,(source + 1 : U32),index,gones)}
    case origins gones:
      NoGones{}

# Every source laid as its ranges in order, and an empty piece with the
# property of its source.
def SourcesStatic(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +source: U32,+fill: Element,models: Models<Element>,gones: Gones) -> Data:
  match origins models gones:
    case Con{+origin,origins} Models{+model,rest} Gones{+gone,gones}:
      Held.Both<{PiecesModel.layout(~Element,model) == origin_ranges(origin) : Pieces.Ranges},Held.Both<Placed.Ordered(origin_ranges(origin)),
        Held.Both<Property(source,0,0,Buffer.content(~Element,empty(~Element,fill))),SourcesStatic(~Element,~Property,origins,(source + 1 : U32),fill,rest,gones)>>>
    case Con{+origin,origins} NoModels{} gones:
      Empty
    case origins models gones:
      Unit

# Every piece component index lists of every source starts a leaf no earlier
# taking reached.
def TakeAll(origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,gones: Gones) -> Data:
  match origins gones:
    case Con{+origin,origins} Gones{+gone,gones}:
      Held.Both<TakeFacts(source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),gone),TakeAll(origins,components,(source + 1 : U32),index,gones)>
    case origins gones:
      Unit

# Every source is good up to its spans taken.
def SourcesGood(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +source: U32,models: Models<Element>,gones: Gones) -> Data:
  match origins models gones:
    case Con{+origin,origins} Models{+model,rest} Gones{+gone,gones}:
      Held.Both<Placed.Good(~Element,~Property(source),model,gone),SourcesGood(~Element,~Property,origins,(source + 1 : U32),rest,gones)>
    case Con{+origin,origins} Models{+model,rest} NoGones{}:
      Empty
    case origins models gones:
      Unit

# Every piece taken holds the values of its source.
def SourcesHeld(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +source: U32,models: Models<Element>) -> Data:
  match origins models:
    case Con{+origin,origins} Models{+model,rest}:
      Held.Both<PiecesModel.ContentAll(~Element,~Property(source),PiecesModel.content(~Element,model)),SourcesHeld(~Element,~Property,origins,(source + 1 : U32),rest)>
    case Con{+origin,origins} NoModels{}:
      Empty
    case origins models:
      Unit

def static_next(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,source,fill,models,gones)) ->
  SourcesStatic(~Element,~Property,origins,source,fill,took_left(~Element,took_model(~Element,origins,components,source,index,fill,models)),gones_after(origins,components,source,index,gones)):
  match origins models gones:
    case Con{+origin,+origins} Models{+model,+rest} Gones{+gone,+gones}:
      Held.Both{+same,+more} = static
      Held.Both{+ordered,+more_static} = more
      Held.Both{+nothing,+later} = more_static
      +pieces = source_pieces(components,source,index,origin)
      +after = taken_model(~Element,pieces,origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      +next_gones = gones_after(origins,components,(source + 1 : U32),index,gones)
      %Equal.sym(Models<Element>,took_left(~Element,took_joined(~Element,after,rest_took)),Models{left_of(~Element,after),took_left(~Element,rest_took)},
          joined_left_models(~Element,after,rest_took)) :
        SourcesStatic(~Element,~Property,Con{origin,origins},source,fill,_,Gones{taken_gone(pieces,origin_grid(origin),gone),next_gones})
      Held.Both{Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,left_of(~Element,after)),PiecesModel.layout(~Element,model),origin_ranges(origin),
          taken_layout(~Element,pieces,origin_grid(origin),origin_ranges(origin),fill,model),same),
        Held.Both{ordered,Held.Both{nothing,static_next(~Element,~Property,origins,components,(source + 1 : U32),index,fill,rest,gones,later)}}}
    case Nil{} +models +gones:
      Unit{}
    case Con{+origin,+origins} NoModels{} +gones:
      Empty.absurd(SourcesStatic(~Element,~Property,Con{origin,origins},source,fill,took_left(~Element,took_model(~Element,Con{origin,origins},components,source,index,fill,NoModels{})),gones_after(Con{origin,origins},components,source,index,gones)),static)
    case Con{+origin,+origins} Models{+model,+rest} NoGones{}:
      +after = taken_model(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Models<Element>,took_left(~Element,took_joined(~Element,after,rest_took)),Models{left_of(~Element,after),took_left(~Element,rest_took)},
          joined_left_models(~Element,after,rest_took)) :
        SourcesStatic(~Element,~Property,Con{origin,origins},source,fill,_,NoGones{})
      Unit{}

def good_next(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,source,fill,models,gones),+take: TakeAll(origins,components,source,index,gones),
  +good: SourcesGood(~Element,~Property,origins,source,models,gones)) ->
  SourcesGood(~Element,~Property,origins,source,took_left(~Element,took_model(~Element,origins,components,source,index,fill,models)),gones_after(origins,components,source,index,gones)):
  match origins models gones:
    case Con{+origin,+origins} Models{+model,+rest} Gones{+gone,+gones}:
      Held.Both{+same,+more} = static
      Held.Both{+ordered,+more_static} = more
      Held.Both{+nothing,+later_static} = more_static
      Held.Both{+facts,+later_take} = take
      Held.Both{+model_good,+later_good} = good
      +pieces = source_pieces(components,source,index,origin)
      +after = taken_model(~Element,pieces,origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      +next_gones = gones_after(origins,components,(source + 1 : U32),index,gones)
      %Equal.sym(Models<Element>,took_left(~Element,took_joined(~Element,after,rest_took)),Models{left_of(~Element,after),took_left(~Element,rest_took)},
          joined_left_models(~Element,after,rest_took)) :
        SourcesGood(~Element,~Property,Con{origin,origins},source,_,Gones{taken_gone(pieces,origin_grid(origin),gone),next_gones})
      Held.Both{taken_good(~Element,~Property(source),after,taken_gone(pieces,origin_grid(origin),gone),
          taken_facts(~Element,~Property(source),pieces,origin_grid(origin),origin_ranges(origin),fill,model,gone,same,ordered,nothing,facts,model_good)),
        good_next(~Element,~Property,origins,components,(source + 1 : U32),index,fill,rest,gones,later_static,later_take,later_good)}
    case Nil{} +models +gones:
      Unit{}
    case Con{+origin,+origins} NoModels{} +gones:
      Empty.absurd(SourcesGood(~Element,~Property,Con{origin,origins},source,took_left(~Element,took_model(~Element,Con{origin,origins},components,source,index,fill,NoModels{})),gones_after(Con{origin,origins},components,source,index,gones)),static)
    case Con{+origin,+origins} Models{+model,+rest} NoGones{}:
      +after = taken_model(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Models<Element>,took_left(~Element,took_joined(~Element,after,rest_took)),Models{left_of(~Element,after),took_left(~Element,rest_took)},
          joined_left_models(~Element,after,rest_took)) :
        SourcesGood(~Element,~Property,Con{origin,origins},source,_,NoGones{})
      good

def held_next(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,origins: List<&2,Tensors.Origin>,
  +components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,source,fill,models,gones),+take: TakeAll(origins,components,source,index,gones),
  +good: SourcesGood(~Element,~Property,origins,source,models,gones)) ->
  SourcesHeld(~Element,~Property,origins,source,took_taken(~Element,took_model(~Element,origins,components,source,index,fill,models))):
  match origins models gones:
    case Con{+origin,+origins} Models{+model,+rest} Gones{+gone,+gones}:
      Held.Both{+same,+more} = static
      Held.Both{+ordered,+more_static} = more
      Held.Both{+nothing,+later_static} = more_static
      Held.Both{+facts,+later_take} = take
      Held.Both{+model_good,+later_good} = good
      +pieces = source_pieces(components,source,index,origin)
      +after = taken_model(~Element,pieces,origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Models<Element>,took_taken(~Element,took_joined(~Element,after,rest_took)),Models{taken_of(~Element,after),took_taken(~Element,rest_took)},
          joined_taken_models(~Element,after,rest_took)) :
        SourcesHeld(~Element,~Property,Con{origin,origins},source,_)
      Held.Both{taken_all(~Element,~Property(source),after,taken_gone(pieces,origin_grid(origin),gone),
          taken_facts(~Element,~Property(source),pieces,origin_grid(origin),origin_ranges(origin),fill,model,gone,same,ordered,nothing,facts,model_good)),
        held_next(~Element,~Property,origins,components,(source + 1 : U32),index,fill,rest,gones,later_static,later_take,later_good)}
    case Nil{} +models +gones:
      Unit{}
    case Con{+origin,+origins} NoModels{} +gones:
      Empty.absurd(SourcesHeld(~Element,~Property,Con{origin,origins},source,took_taken(~Element,took_model(~Element,Con{origin,origins},components,source,index,fill,NoModels{}))),static)
    case Con{+origin,+origins} Models{+model,+rest} NoGones{}:
      +after = taken_model(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Models<Element>,took_taken(~Element,took_joined(~Element,after,rest_took)),Models{taken_of(~Element,after),took_taken(~Element,rest_took)},
          joined_taken_models(~Element,after,rest_took)) :
        SourcesHeld(~Element,~Property,Con{origin,origins},source,_)
      Empty.absurd(SourcesHeld(~Element,~Property,Con{origin,origins},source,Models{taken_of(~Element,after),took_taken(~Element,rest_took)}),good)

# ---- Every component of a layout ----

# The spans of every source after the components of a layout took theirs.
def layout_gones(layout: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+gones: Gones) -> Gones:
  match layout:
    case Pieces.Range{+first,count}:
      gones_after(origins,components,0,first,gones)
    case Pieces.Split{left,right}:
      layout_gones(right,origins,components,layout_gones(left,origins,components,gones))

# Every component of a layout lists pieces no earlier component took.
def BuildReady(layout: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+gones: Gones) -> Data:
  match layout:
    case Pieces.Range{+first,count}:
      Held.Both<TakeAll(origins,components,0,first,gones),{Nat.is_lt(U32.to_nat(first),U32.to_nat(Tasks.components_count(components))) == True{} : Bool}>
    case Pieces.Split{+left,+right}:
      Held.Both<BuildReady(left,origins,components,gones),BuildReady(right,origins,components,layout_gones(left,origins,components,gones))>

# The spans of the pieces taken in order, as the stack of the pieces taken
# lists them.
def taken_spans(pieces: List<&2,U32>,+grid: Grid.Grid) -> Held.Spans:
  match pieces:
    case Nil{}:
      Held.Spanned{0,0,Held.Ended{}}
    case Con{+piece,Nil{}}:
      Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),Held.Ended{}}
    case Con{+piece,rest}:
      Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),taken_spans(rest,grid)}

def joined_taken(~Element: Data,+leaf: PiecesModel.Model<Element>,+after: Taken<Element>) ->
  {taken_of(~Element,joined(~Element,leaf,after)) == PiecesModel.Fork{leaf,taken_of(~Element,after)} : PiecesModel.Model<Element>}:
  Taken{+rest,+taken} = after
  {==}

def taken_laid(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,+model: PiecesModel.Model<Element>) ->
  {Held.spans(~Element,Held.stacked_model(~Element,taken_of(~Element,taken_model(~Element,pieces,grid,layout,fill,model)),Held.Bottom{})) == taken_spans(pieces,grid) : Held.Spans}:
  match pieces:
    case Nil{}:
      {==}
    case Con{+piece,Nil{}}:
      {==}
    case Con{+piece,Con{+next,+more}}:
      +leaf = taken_leaf(~Element,piece,grid,layout,fill,model)
      +left = taken_rest(~Element,piece,grid,layout,fill,model)
      +after = taken_model(~Element,Con{next,more},grid,layout,fill,left)
      %Equal.sym(PiecesModel.Model<Element>,taken_of(~Element,joined(~Element,leaf,after)),PiecesModel.Fork{leaf,taken_of(~Element,after)},joined_taken(~Element,leaf,after)) :
        {Held.spans(~Element,Held.stacked_model(~Element,_,Held.Bottom{})) == taken_spans(Con{piece,Con{next,more}},grid) : Held.Spans}
      Equal.cong(Held.Spans,Held.Spans,rest => Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),rest},
        Held.spans(~Element,Held.stacked_model(~Element,taken_of(~Element,after),Held.Bottom{})),taken_spans(Con{next,more},grid),
        taken_laid(~Element,Con{next,more},grid,layout,fill,left))

# The pieces every source holds for component index are the pieces it lists.
def SourcesLaid(~Element: Data,origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,models: Models<Element>) -> Data:
  match origins models:
    case Con{+origin,origins} Models{+model,rest}:
      Held.Both<{Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == taken_spans(source_pieces(components,source,index,origin),origin_grid(origin)) : Held.Spans},
        SourcesLaid(~Element,origins,components,(source + 1 : U32),index,rest)>
    case origins models:
      Unit

def laid_next(~Element: Data,origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+source: U32,+index: U32,+fill: Element,+models: Models<Element>) ->
  SourcesLaid(~Element,origins,components,source,index,took_taken(~Element,took_model(~Element,origins,components,source,index,fill,models))):
  match origins models:
    case Con{+origin,+origins} Models{+model,+rest}:
      +after = taken_model(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model)
      +rest_took = took_model(~Element,origins,components,(source + 1 : U32),index,fill,rest)
      %Equal.sym(Models<Element>,took_taken(~Element,took_joined(~Element,after,rest_took)),Models{taken_of(~Element,after),took_taken(~Element,rest_took)},
          joined_taken_models(~Element,after,rest_took)) :
        SourcesLaid(~Element,Con{origin,origins},components,source,index,_)
      Held.Both{taken_laid(~Element,source_pieces(components,source,index,origin),origin_grid(origin),origin_ranges(origin),fill,model),
        laid_next(~Element,origins,components,(source + 1 : U32),index,fill,rest)}
    case Nil{} +models:
      Unit{}
    case Con{+origin,+origins} NoModels{}:
      Unit{}

# The pieces component index took of every source hold the values of their
# source and are the pieces it lists.
def TakenReady(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+models: Models<Element>,+index: U32) -> Data:
  Held.Both<SourcesHeld(~Element,~Property,origins,0,models),SourcesLaid(~Element,origins,components,0,index,models)>

def ComponentHeld(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+part: Models<Element>,+first: U32,+rows: U32) -> Data:
  Held.Both<TakenReady(~Element,~Property,components,origins,part,first),{Nat.is_lt(U32.to_nat(first),U32.to_nat(Tasks.components_count(components))) == True{} : Bool}>

def static_built(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,layout: Pieces.Ranges,
  +origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,0,fill,models,gones)) ->
  SourcesStatic(~Element,~Property,origins,0,fill,built_left(~Element,built_model(~Element,layout,origins,components,fill,models)),layout_gones(layout,origins,components,gones)):
  match layout:
    case Pieces.Range{+first,+count}:
      static_next(~Element,~Property,origins,components,0,first,fill,models,gones,static)
    case Pieces.Split{+left,+right}:
      static_built(~Element,~Property,right,origins,components,fill,built_left(~Element,built_model(~Element,left,origins,components,fill,models)),
        layout_gones(left,origins,components,gones),static_built(~Element,~Property,left,origins,components,fill,models,gones,static))

def good_built(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,layout: Pieces.Ranges,
  +origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,0,fill,models,gones),+ready: BuildReady(layout,origins,components,gones),
  +good: SourcesGood(~Element,~Property,origins,0,models,gones)) ->
  SourcesGood(~Element,~Property,origins,0,built_left(~Element,built_model(~Element,layout,origins,components,fill,models)),layout_gones(layout,origins,components,gones)):
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+take,+inside} = ready
      good_next(~Element,~Property,origins,components,0,first,fill,models,gones,static,take,good)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_ready,+right_ready} = ready
      good_built(~Element,~Property,right,origins,components,fill,built_left(~Element,built_model(~Element,left,origins,components,fill,models)),
        layout_gones(left,origins,components,gones),static_built(~Element,~Property,left,origins,components,fill,models,gones,static),right_ready,
        good_built(~Element,~Property,left,origins,components,fill,models,gones,static,left_ready,good))

# Every component took pieces that hold the values of their sources.
def held_built(~Element: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,layout: Pieces.Ranges,
  +origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+fill: Element,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,0,fill,models,gones),+ready: BuildReady(layout,origins,components,gones),
  +good: SourcesGood(~Element,~Property,origins,0,models,gones)) ->
  PiecesModel.ShapedAll(~Models<Element>,~List<&2,Tensors.Origin>,~ComponentHeld(~Element,~Property,components),origins,built_inputs(~Element,built_model(~Element,layout,origins,components,fill,models))):
  match layout:
    case Pieces.Range{+first,+count}:
      Held.Both{+take,+inside} = ready
      Held.Both{Held.Both{held_next(~Element,~Property,origins,components,0,first,fill,models,gones,static,take,good),laid_next(~Element,origins,components,0,first,fill,models)},
        inside}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_ready,+right_ready} = ready
      KernelGemm.Both{held_built(~Element,~Property,left,origins,components,fill,models,gones,static,left_ready,good),
        held_built(~Element,~Property,right,origins,components,fill,built_left(~Element,built_model(~Element,left,origins,components,fill,models)),
          layout_gones(left,origins,components,gones),static_built(~Element,~Property,left,origins,components,fill,models,gones,static),right_ready,
          good_built(~Element,~Property,left,origins,components,fill,models,gones,static,left_ready,good))}

# ---- Gathering the target pieces of a component ----

# The holding of every source a task reads.
type Holdings<-Element: Data> is Data:
  Unheld{}
  Holds{holding: Held.Holding<Element>,rest: Holdings<Element>}

def pack_holdings(~Element: Data,holdings: Holdings<Element>) -> Tensors.Holdings<Element>:
  match holdings:
    case Unheld{}:
      Tensors.Unheld{}
    case Holds{holding,rest}:
      Tensors.Holds{Held.pack(~Element,holding),pack_holdings(~Element,rest)}

# Every source held from its first piece.
def held_sources(~Element: Data,models: Models<Element>) -> Holdings<Element>:
  match models:
    case NoModels{}:
      Unheld{}
    case Models{model,rest}:
      Holds{Held.Holding{Held.Bottom{},Held.stacked_model(~Element,model,Held.Bottom{})},held_sources(~Element,rest)}

def held_exact(~Element: Data,+models: Models<Element>) ->
  {Tensors.holdings(~Element,pack_sources(~Element,models)) == pack_holdings(~Element,held_sources(~Element,models)) : Tensors.Holdings<Element>}:
  match models:
    case NoModels{}:
      {==}
    case Models{+model,+rest}:
      %Held.stacked_packed(~Element,model,Held.Bottom{}) :
        {Tensors.Holds{Move.Holding{Move.Bottom{},Move.stacked(~Element,PiecesModel.pack(~Element,model),Move.Bottom{})},Tensors.holdings(~Element,pack_sources(~Element,rest))} ==
          Tensors.Holds{Move.Holding{Move.Bottom{},_},pack_holdings(~Element,held_sources(~Element,rest))} : Tensors.Holdings<Element>}
      %held_exact(~Element,rest) :
        {Tensors.Holds{Move.Holding{Move.Bottom{},Move.stacked(~Element,PiecesModel.pack(~Element,model),Move.Bottom{})},Tensors.holdings(~Element,pack_sources(~Element,rest))} == Tensors.Holds{Move.Holding{Move.Bottom{},Move.stacked(~Element,PiecesModel.pack(~Element,model),Move.Bottom{})},_} : Tensors.Holdings<Element>}
      {==}

# What a task holds after a gather and the target piece it wrote.
type Gathered<-Element: Data> is Data:
  Gathered{holdings: Holdings<Element>,output: Buffer.Model<Element>}

def gathered_holdings(~Element: Data,+gathered: Gathered<Element>) -> Holdings<Element>:
  Gathered{+holdings,+output} = gathered
  holdings

def gathered_output(~Element: Data,+gathered: Gathered<Element>) -> Buffer.Model<Element>:
  Gathered{+holdings,+output} = gathered
  output

# A model of a gather of count cells from first.
def GatherModel(~Element: Data,~Context: Data) -> Type:
  @+context: Context -> @+components: Tasks.Components -> @+first: U32 -> @+count: U32 -> @+fill: Element -> @+holdings: Holdings<Element> -> Gathered<Element>

# Every holding orders the pieces taken of its source.
def HeldOrder(~Element: Data,holdings: Holdings<Element>,models: Models<Element>) -> Data:
  match holdings models:
    case Holds{+holding,rest} Models{+model,more}:
      Held.Both<{Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},HeldOrder(~Element,rest,more)>
    case Unheld{} NoModels{}:
      Unit
    case holdings models:
      Empty

def held_ordered(~Element: Data,+models: Models<Element>) -> HeldOrder(~Element,held_sources(~Element,models),models):
  match models:
    case NoModels{}:
      Unit{}
    case Models{+model,+rest}:
      Held.Both{{==},held_ordered(~Element,rest)}

# A piece of a list, and pieces of a list.
def Member(pieces: List<&2,U32>,+piece: U32) -> Data:
  match pieces:
    case Nil{}:
      Empty
    case Con{+first,rest}:
      Placed.Side<{first == piece : U32},Member(rest,piece)>

def Members(pieces: List<&2,U32>,+whole: List<&2,U32>) -> Data:
  match pieces:
    case Nil{}:
      Unit
    case Con{+piece,rest}:
      Held.Both<Member(whole,piece),Members(rest,whole)>

def members_later(pieces: List<&2,U32>,+first: U32,+whole: List<&2,U32>,+members: Members(pieces,whole)) -> Members(pieces,Con{first,whole}):
  match pieces:
    case Nil{}:
      Unit{}
    case Con{+piece,+rest}:
      Held.Both{+member,+later} = members
      Held.Both{Placed.OnRight{member},members_later(rest,first,whole,later)}

def members_all(pieces: List<&2,U32>) -> Members(pieces,pieces):
  match pieces:
    case Nil{}:
      Unit{}
    case Con{+piece,+rest}:
      Held.Both{Placed.OnLeft{{==}},members_later(rest,piece,rest,members_all(rest))}

# A gather writes the target piece with the target property and keeps the
# holdings in order.
def GatherFacts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Holdings<Element>,+models: Models<Element>,+gathered: Gathered<Element>) -> Data:
  Held.Both<{gather(context,components,first,count,fill,pack_holdings(~Element,holdings)) ==
      (pack_holdings(~Element,gathered_holdings(~Element,gathered)),Buffer.pack(~Element,gathered_output(~Element,gathered))) : Tensors.Holdings<Element> & Storage.Buffer<Element>},
    Held.Both<HeldOrder(~Element,gathered_holdings(~Element,gathered),models),Target(first,count,Buffer.content(~Element,gathered_output(~Element,gathered)))>>

# The law of a gather: a target piece of component index, gathered in a
# context that agrees with the components, the sources, the target and the
# fill, from
# holdings that order the pieces it took of every source.
def GatherLaw(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data) -> Type:
  @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> @+context: Context ->
  @+index: U32 -> @+piece: U32 -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+agrees: Agrees(context,components,origins,target,target_grid,fill) ->
  @+ready: TakenReady(~Element,~Property,components,origins,models,index) -> @+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool} ->
  @+order: HeldOrder(~Element,holdings,models) -> @+member: Member(Tasks.target_pieces(components,target_grid,target,index),piece) ->
  GatherFacts(~Element,~Context,~gather,~Target,context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings,models,
    gather_model(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings))

# What a task keeps from one target piece of a component to the next: Kept of
# the target grid, the models of the sources, the holdings and the target
# pieces still to gather. A gather of a target piece writes it with the target
# property and keeps Kept for the pieces after it.
def KeptFacts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  +context: Context,+components: Tasks.Components,+target_grid: Grid.Grid,+piece: U32,+rest: List<&2,U32>,+fill: Element,+holdings: Holdings<Element>,
  +models: Models<Element>,+gathered: Gathered<Element>) -> Data:
  Held.Both<{gather(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,pack_holdings(~Element,holdings)) ==
      (pack_holdings(~Element,gathered_holdings(~Element,gathered)),Buffer.pack(~Element,gathered_output(~Element,gathered))) : Tensors.Holdings<Element> & Storage.Buffer<Element>},
    Held.Both<Kept(target_grid,models,gathered_holdings(~Element,gathered),rest),
      Target(Grid.start(target_grid,piece),Grid.span(target_grid,piece),Buffer.content(~Element,gathered_output(~Element,gathered)))>>

# The law of a gather that keeps Kept: a target piece of component index, the
# pieces after it still to gather, from holdings that keep Kept for them all.
def KeptLaw(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data) -> Type:
  @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> @+context: Context ->
  @+index: U32 -> @+piece: U32 -> @+rest: List<&2,U32> -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+agrees: Agrees(context,components,origins,target,target_grid,fill) ->
  @+ready: TakenReady(~Element,~Property,components,origins,models,index) -> @+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool} ->
  @+kept: Kept(target_grid,models,holdings,Con{piece,rest}) -> @+member: Member(Tasks.target_pieces(components,target_grid,target,index),piece) ->
  @+after: NextAfter(target_grid,piece,rest) ->
  KeptFacts(~Element,~Context,~gather,~Target,~Kept,context,components,target_grid,piece,rest,fill,holdings,models,
    gather_model(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings))

# Kept of the sources held from their first pieces, before every target piece
# of a component.
def KeptBegin(~Element: Data,~Context: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data) -> Type:
  @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> @+context: Context ->
  @+layout: Pieces.Ranges -> @+index: U32 -> @+models: Models<Element> -> @+agrees: Agrees(context,components,origins,target,target_grid,fill) ->
  @+ready: TakenReady(~Element,~Property,components,origins,models,index) ->
  @+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool} ->
  @+targets: TargetsAt(layout,target_grid,Tasks.target_pieces(components,target_grid,target,index)) ->
  Kept(target_grid,models,held_sources(~Element,models),Tasks.target_pieces(components,target_grid,target,index))

# A gather law keeps the holdings in order.
def ordered_kept(~Element: Data,+grid: Grid.Grid,+models: Models<Element>,+holdings: Holdings<Element>,+pieces: List<&2,U32>) -> Data:
  HeldOrder(~Element,holdings,models)

def ordered_law(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  gathers: GatherLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees)) ->
  KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~ordered_kept(~Element)):
  components => origins => target => target_grid => fill => context => index => piece => rest => models => holdings => agrees => ready => inside => kept => member => after =>
    gathers(components,origins,target,target_grid,fill,context,index,piece,models,holdings,agrees,ready,inside,kept,member)

def ordered_begin(~Element: Data,~Context: Data,~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data) ->
  KeptBegin(~Element,~Context,~Property,~Agrees,~ordered_kept(~Element)):
  components => origins => target => target_grid => fill => context => layout => index => models => agrees => ready => inside => targets =>
    held_ordered(~Element,models)

# A target piece starts a leaf of the target layout with its cells and is not
# empty, and a component lists at least one, each after the end of the one
# before.
def TargetAt(+layout: Pieces.Ranges,+grid: Grid.Grid,+piece: U32) -> Data:
  Held.Both<Placed.LeafAt(layout,Grid.start(grid,piece)),
    Held.Both<{Placed.rows_of(layout,Grid.start(grid,piece)) == Grid.span(grid,piece) : U32},{U32.is_zero(Grid.span(grid,piece)) == False{} : Bool}>>

def NextAfter(+grid: Grid.Grid,+piece: U32,rest: List<&2,U32>) -> Data:
  match rest:
    case Nil{}:
      Unit
    case Con{+next,more}:
      {Nat.is_le(Nat.add(U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece))),U32.to_nat(Grid.start(grid,next))) == True{} : Bool}

def TargetsAt(+layout: Pieces.Ranges,+grid: Grid.Grid,pieces: List<&2,U32>) -> Data:
  match pieces:
    case Nil{}:
      Empty
    case Con{+piece,Nil{}}:
      TargetAt(layout,grid,piece)
    case Con{+piece,rest}:
      Held.Both<TargetAt(layout,grid,piece),Held.Both<NextAfter(grid,piece,rest),TargetsAt(layout,grid,rest)>>

# What a task holds after gathering its target pieces, and the pieces.
type GatheredPieces<-Element: Data> is Data:
  GatheredPieces{holdings: Holdings<Element>,pieces: PiecesModel.Model<Element>}

def pieces_holdings(~Element: Data,+gathered: GatheredPieces<Element>) -> Holdings<Element>:
  GatheredPieces{+holdings,+pieces} = gathered
  holdings

def pieces_of(~Element: Data,+gathered: GatheredPieces<Element>) -> PiecesModel.Model<Element>:
  GatheredPieces{+holdings,+pieces} = gathered
  pieces

# The target pieces gathered in order, as Tensors.gathered_pieces gathers them.
def gathered_model(~Element: Data,~Context: Data,~gather_model: GatherModel(~Element,~Context),pieces: List<&2,U32>,+context: Context,+components: Tasks.Components,
  +target_grid: Grid.Grid,+fill: Element,+holdings: Holdings<Element>) -> GatheredPieces<Element>:
  match pieces:
    case Nil{}:
      GatheredPieces{holdings,PiecesModel.Leaf{empty(~Element,fill),0,0}}
    case Con{+piece,Nil{}}:
      +first = Grid.start(target_grid,piece)
      +count = Grid.span(target_grid,piece)
      +step = gather_model(context,components,first,count,fill,holdings)
      GatheredPieces{gathered_holdings(~Element,step),PiecesModel.Leaf{gathered_output(~Element,step),first,count}}
    case Con{+piece,rest}:
      +first = Grid.start(target_grid,piece)
      +count = Grid.span(target_grid,piece)
      +step = gather_model(context,components,first,count,fill,holdings)
      +after = gathered_model(~Element,~Context,~gather_model,rest,context,components,target_grid,fill,gathered_holdings(~Element,step))
      GatheredPieces{pieces_holdings(~Element,after),PiecesModel.Fork{PiecesModel.Leaf{gathered_output(~Element,step),first,count},pieces_of(~Element,after)}}

def GatheredFacts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  +layout: Pieces.Ranges,pieces: List<&2,U32>,+context: Context,+components: Tasks.Components,+target_grid: Grid.Grid,+fill: Element,+holdings: Holdings<Element>,
  +lefts: List<&2,PiecesModel.Model<Element>>,+models: Models<Element>,+gathered: GatheredPieces<Element>) -> Data:
  Held.Both<{Tensors.gathered_pieces(~Element,~Context,~gather,pieces,context,components,target_grid,fill,
        Pieces.Walking{Each.pack_lefts(~Element,lefts),pack_holdings(~Element,holdings)}) ==
      Pieces.Walked{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),Each.pack_lefts(~Element,lefts),pack_holdings(~Element,pieces_holdings(~Element,gathered))} :
      Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>},
    Held.Both<Kept(target_grid,models,pieces_holdings(~Element,gathered),Nil{}),Placed.PiecesAt(~Element,~Target,layout,pieces_of(~Element,gathered))>>

def facts_order(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  +context: Context,+components: Tasks.Components,+target_grid: Grid.Grid,+piece: U32,+rest: List<&2,U32>,+fill: Element,+holdings: Holdings<Element>,
  +models: Models<Element>,+gathered: Gathered<Element>,
  +facts: KeptFacts(~Element,~Context,~gather,~Target,~Kept,context,components,target_grid,piece,rest,fill,holdings,models,gathered)) ->
  Kept(target_grid,models,gathered_holdings(~Element,gathered),rest):
  Held.Both{+same,+kept} = facts
  Held.Both{+order,+holds} = kept
  order

# The holdings and the buffer of a gather, read into a target piece.
def leaf_gathered(~Element: Data,+lefts: List<&2,PiecesModel.Model<Element>>,+first: U32,+count: U32,-pair: Tensors.Holdings<Element> & Storage.Buffer<Element>,+holdings: Holdings<Element>,
  +output: Buffer.Model<Element>,+same: {pair == (pack_holdings(~Element,holdings),Buffer.pack(~Element,output)) : Tensors.Holdings<Element> & Storage.Buffer<Element>}) ->
  {Tensors.gathered_walked(~Element,Each.pack_lefts(~Element,lefts),first,count,pair) ==
    Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Leaf{output,first,count}),Each.pack_lefts(~Element,lefts),pack_holdings(~Element,holdings)} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}:
  %Equal.sym(Tensors.Holdings<Element> & Storage.Buffer<Element>,pair,(pack_holdings(~Element,holdings),Buffer.pack(~Element,output)),same) :
    {Tensors.gathered_walked(~Element,Each.pack_lefts(~Element,lefts),first,count,_) ==
      Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Leaf{output,first,count}),Each.pack_lefts(~Element,lefts),pack_holdings(~Element,holdings)} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}
  {==}

def gathered_one(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  +layout: Pieces.Ranges,+piece: U32,+context: Context,+components: Tasks.Components,+target_grid: Grid.Grid,+fill: Element,+holdings: Holdings<Element>,
  +lefts: List<&2,PiecesModel.Model<Element>>,+models: Models<Element>,+step: Gathered<Element>,
  +facts: KeptFacts(~Element,~Context,~gather,~Target,~Kept,context,components,target_grid,piece,Nil{},fill,holdings,models,step),
  +at: TargetAt(layout,target_grid,piece)) ->
  GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,[piece],context,components,target_grid,fill,holdings,lefts,models,
    GatheredPieces{gathered_holdings(~Element,step),PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)}}):
  Held.Both{+same,+kept} = facts
  Held.Both{+order,+holds} = kept
  Held.Both{+leaf,+sized} = at
  Held.Both{+cells,+filled} = sized
  +first = Grid.start(target_grid,piece)
  +count = Grid.span(target_grid,piece)
  Held.Both{leaf_gathered(~Element,lefts,first,count,gather(context,components,first,count,fill,pack_holdings(~Element,holdings)),gathered_holdings(~Element,step),
      gathered_output(~Element,step),same),
    Held.Both{order,Placed.PieceAt{leaf,cells,filled,holds}}}

# A gathered piece waits on the left for the pieces after it, and joins them.
def more_gathered(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),+piece: U32,+next: U32,+more: List<&2,U32>,+context: Context,
  +components: Tasks.Components,+target_grid: Grid.Grid,+fill: Element,+holdings: Holdings<Element>,+lefts: List<&2,PiecesModel.Model<Element>>,+step: Gathered<Element>,+after: GatheredPieces<Element>,
  +same: {gather(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,pack_holdings(~Element,holdings)) ==
    (pack_holdings(~Element,gathered_holdings(~Element,step)),Buffer.pack(~Element,gathered_output(~Element,step))) : Tensors.Holdings<Element> & Storage.Buffer<Element>},
  +later: {Tensors.gathered_pieces(~Element,~Context,~gather,Con{next,more},context,components,target_grid,fill,
      Pieces.Walking{Each.pack_lefts(~Element,Con{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},lefts}),
        pack_holdings(~Element,gathered_holdings(~Element,step))}) ==
    Pieces.Walked{PiecesModel.pack(~Element,pieces_of(~Element,after)),
      Each.pack_lefts(~Element,Con{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},lefts}),
      pack_holdings(~Element,pieces_holdings(~Element,after))} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}) ->
  {Tensors.gathered_pieces(~Element,~Context,~gather,Con{piece,Con{next,more}},context,components,target_grid,fill,
      Pieces.Walking{Each.pack_lefts(~Element,lefts),pack_holdings(~Element,holdings)}) ==
    Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Fork{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},
        pieces_of(~Element,after)}),Each.pack_lefts(~Element,lefts),pack_holdings(~Element,pieces_holdings(~Element,after))} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}:
  +first = Grid.start(target_grid,piece)
  +count = Grid.span(target_grid,piece)
  %Equal.sym(Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>,
      Tensors.gathered_walked(~Element,Each.pack_lefts(~Element,lefts),first,count,gather(context,components,first,count,fill,pack_holdings(~Element,holdings))),
      Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Leaf{gathered_output(~Element,step),first,count}),Each.pack_lefts(~Element,lefts),
        pack_holdings(~Element,gathered_holdings(~Element,step))},
      leaf_gathered(~Element,lefts,first,count,gather(context,components,first,count,fill,pack_holdings(~Element,holdings)),gathered_holdings(~Element,step),
        gathered_output(~Element,step),same)) :
    {Pieces.joined(~Storage.Buffer<Element>,~Tensors.Holdings<Element>,Tensors.gathered_pieces(~Element,~Context,~gather,Con{next,more},context,components,target_grid,fill,
        Pieces.pushed(~Storage.Buffer<Element>,~Tensors.Holdings<Element>,_))) ==
      Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Fork{PiecesModel.Leaf{gathered_output(~Element,step),first,count},pieces_of(~Element,after)}),
        Each.pack_lefts(~Element,lefts),pack_holdings(~Element,pieces_holdings(~Element,after))} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}
  %Equal.sym(Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>,
      Tensors.gathered_pieces(~Element,~Context,~gather,Con{next,more},context,components,target_grid,fill,
        Pieces.Walking{Each.pack_lefts(~Element,Con{PiecesModel.Leaf{gathered_output(~Element,step),first,count},lefts}),pack_holdings(~Element,gathered_holdings(~Element,step))}),
      Pieces.Walked{PiecesModel.pack(~Element,pieces_of(~Element,after)),Each.pack_lefts(~Element,Con{PiecesModel.Leaf{gathered_output(~Element,step),first,count},lefts}),pack_holdings(~Element,pieces_holdings(~Element,after))},later) :
    {Pieces.joined(~Storage.Buffer<Element>,~Tensors.Holdings<Element>,_) ==
      Pieces.Walked{PiecesModel.pack(~Element,PiecesModel.Fork{PiecesModel.Leaf{gathered_output(~Element,step),first,count},pieces_of(~Element,after)}),
        Each.pack_lefts(~Element,lefts),pack_holdings(~Element,pieces_holdings(~Element,after))} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}
  {==}

def gathered_more(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  +layout: Pieces.Ranges,+piece: U32,+next: U32,+more: List<&2,U32>,+context: Context,+components: Tasks.Components,+target_grid: Grid.Grid,+fill: Element,
  +holdings: Holdings<Element>,+lefts: List<&2,PiecesModel.Model<Element>>,+models: Models<Element>,+step: Gathered<Element>,
  +facts: KeptFacts(~Element,~Context,~gather,~Target,~Kept,context,components,target_grid,piece,Con{next,more},fill,holdings,models,step),
  +at: TargetAt(layout,target_grid,piece),
  +later: GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step),
    Con{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},lefts},models,
    gathered_model(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step)))) ->
  GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,Con{piece,Con{next,more}},context,components,target_grid,fill,holdings,lefts,models,
    GatheredPieces{pieces_holdings(~Element,gathered_model(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step))),
      PiecesModel.Fork{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},
        pieces_of(~Element,gathered_model(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step)))}}):
  Held.Both{+same,+kept} = facts
  Held.Both{+order,+holds} = kept
  Held.Both{+leaf,+sized} = at
  Held.Both{+cells,+filled} = sized
  Held.Both{+later_same,+later_kept} = later
  Held.Both{+later_order,+later_pieces} = later_kept
  +after = gathered_model(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step))
  Held.Both{more_gathered(~Element,~Context,~gather,piece,next,more,context,components,target_grid,fill,holdings,lefts,step,after,same,later_same),
    Held.Both{later_order,Held.Both{Placed.PieceAt{leaf,cells,filled,holds},later_pieces}}}

# Gathering the target pieces a component lists, in order, writes each with
# the target property and keeps Kept.
def gathered_facts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +layout: Pieces.Ranges,pieces: List<&2,U32>,+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,
  +fill: Element,+context: Context,+agrees: Agrees(context,components,origins,target,target_grid,fill),+index: U32,+models: Models<Element>,+holdings: Holdings<Element>,
  +lefts: List<&2,PiecesModel.Model<Element>>,+ready: TakenReady(~Element,~Property,components,origins,models,index),+inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},
  +kept: Kept(target_grid,models,holdings,pieces),+members: Members(pieces,Tasks.target_pieces(components,target_grid,target,index)),+targets: TargetsAt(layout,target_grid,pieces)) ->
  GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,pieces,context,components,target_grid,fill,holdings,lefts,models,
    gathered_model(~Element,~Context,~gather_model,pieces,context,components,target_grid,fill,holdings)):
  match pieces:
    case Nil{}:
      Empty.absurd(GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,Nil{},context,components,target_grid,fill,holdings,lefts,models,
        gathered_model(~Element,~Context,~gather_model,Nil{},context,components,target_grid,fill,holdings)),targets)
    case Con{+piece,Nil{}}:
      Held.Both{+member,+rest_members} = members
      gathered_one(~Element,~Context,~gather,~Target,~Kept,layout,piece,context,components,target_grid,fill,holdings,lefts,models,
        gather_model(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings),
        gathers(components,origins,target,target_grid,fill,context,index,piece,Nil{},models,holdings,agrees,ready,inside,kept,member,Unit{}),targets)
    case Con{+piece,Con{+next,+more}}:
      Held.Both{+member,+rest_members} = members
      Held.Both{+at,+later_targets} = targets
      Held.Both{+after,+rest_targets} = later_targets
      +step = gather_model(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings)
      +facts = gathers(components,origins,target,target_grid,fill,context,index,piece,Con{next,more},models,holdings,agrees,ready,inside,kept,member,after)
      gathered_more(~Element,~Context,~gather,~gather_model,~Target,~Kept,layout,piece,next,more,context,components,target_grid,fill,holdings,lefts,models,step,facts,at,
        gathered_facts(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,Con{next,more},components,origins,target,target_grid,fill,context,agrees,index,
          models,gathered_holdings(~Element,step),
          Con{PiecesModel.Leaf{gathered_output(~Element,step),Grid.start(target_grid,piece),Grid.span(target_grid,piece)},lefts},ready,inside,
          facts_order(~Element,~Context,~gather,~Target,~Kept,context,components,target_grid,piece,Con{next,more},fill,holdings,models,step,facts),
          rest_members,rest_targets))

# ---- Every component ----

# The target pieces of component index, gathered from the pieces it took.
def component_out(~Element: Data,~Context: Data,~gather_model: GatherModel(~Element,~Context),+components: Tasks.Components,+target: List<&2,U32>,
  +target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>,+index: U32) -> PiecesModel.Model<Element>:
  pieces_of(~Element,gathered_model(~Element,~Context,~gather_model,Tasks.target_pieces(components,target_grid,target,index),context,components,target_grid,fill,
    held_sources(~Element,models)))

def ComponentRun(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>,+first: U32,+count: U32) -> Data:
  Held.Both<{Tensors.component_run(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},first,count,pack_sources(~Element,models),Unit{}) ==
      Partition.Leaf{PiecesModel.pack(~Element,component_out(~Element,~Context,~gather_model,components,target,target_grid,fill,context,models,first)),first,count,Unit{}} :
      Partition.Tree<&1,Pieces.Pieces(~Element),Unit>},
    Placed.PiecesAt(~Element,~Target,layout,component_out(~Element,~Context,~gather_model,components,target,target_grid,fill,context,models,first))>

def run_same(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,
  +target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>,+first: U32,+count: U32,+gathered: GatheredPieces<Element>,
  +same: {Tensors.gathered_pieces(~Element,~Context,~gather,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
      Pieces.Walking{Each.pack_lefts(~Element,Nil{}),pack_holdings(~Element,held_sources(~Element,models))}) ==
    Pieces.Walked{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),Each.pack_lefts(~Element,Nil{}),pack_holdings(~Element,pieces_holdings(~Element,gathered))} : Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>}) ->
  {Tensors.component_run(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},first,count,pack_sources(~Element,models),Unit{}) ==
    Partition.Leaf{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),first,count,Unit{}} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}:
  %Equal.sym(Tensors.Holdings<Element>,Tensors.holdings(~Element,pack_sources(~Element,models)),pack_holdings(~Element,held_sources(~Element,models)),held_exact(~Element,models)) :
    {Partition.Leaf{Tensors.walked_pieces(~Element,Tensors.gathered_pieces(~Element,~Context,~gather,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
        Pieces.Walking{Pieces.NoLefts{},_})),first,count,Unit{}} == Partition.Leaf{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),first,count,Unit{}} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}
  %Equal.sym(Pieces.Walked<Storage.Buffer<Element>,Tensors.Holdings<Element>>,Tensors.gathered_pieces(~Element,~Context,~gather,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
        Pieces.Walking{Each.pack_lefts(~Element,Nil{}),pack_holdings(~Element,held_sources(~Element,models))}),
      Pieces.Walked{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),Each.pack_lefts(~Element,Nil{}),pack_holdings(~Element,pieces_holdings(~Element,gathered))},same) :
    {Partition.Leaf{Tensors.walked_pieces(~Element,_),first,count,Unit{}} == Partition.Leaf{PiecesModel.pack(~Element,pieces_of(~Element,gathered)),first,count,Unit{}} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}
  {==}

def run_from(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,+layout: Pieces.Ranges,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>,+first: U32,+count: U32,
  +facts: GatheredFacts(~Element,~Context,~gather,~Target,~Kept,layout,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
    held_sources(~Element,models),Nil{},models,
    gathered_model(~Element,~Context,~gather_model,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,held_sources(~Element,models)))) ->
  ComponentRun(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,models,first,count):
  Held.Both{+same,+kept} = facts
  Held.Both{+order,+pieces_at} = kept
  +gathered = gathered_model(~Element,~Context,~gather_model,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
    held_sources(~Element,models))
  Held.Both{run_same(~Element,~Context,~gather,components,origins,target,target_grid,fill,context,models,first,count,gathered,same),pieces_at}

# A component whose taken pieces hold the values of their sources writes its
# target pieces with the target property.
def component_exact(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +layout: Pieces.Ranges,+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+agrees: Agrees(context,components,origins,target,target_grid,fill),
  +models: Models<Element>,+first: U32,+count: U32,+ready: TakenReady(~Element,~Property,components,origins,models,first),+inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},
  +targets: TargetsAt(layout,target_grid,Tasks.target_pieces(components,target_grid,target,first))) ->
  ComponentRun(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,models,first,count):
  +pieces = Tasks.target_pieces(components,target_grid,target,first)
  run_from(~Element,~Context,~gather,~gather_model,~Target,~Kept,layout,components,origins,target,target_grid,fill,context,models,first,count,
    gathered_facts(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,pieces,components,origins,target,target_grid,fill,context,agrees,first,models,
      held_sources(~Element,models),Nil{},ready,inside,begin(components,origins,target,target_grid,fill,context,layout,first,models,agrees,ready,inside,targets),members_all(pieces),targets))

# ---- Every component of the tasks ----

# The target pieces every component of a tree gathers, a task per component.
def shaped_results(~Element: Data,~Context: Data,~gather_model: GatherModel(~Element,~Context),+components: Tasks.Components,+target: List<&2,U32>,
  +target_grid: Grid.Grid,+fill: Element,+context: Context,tree: PiecesModel.Shaped<Models<Element>>) -> Partition.Tree<&2,PiecesModel.Model<Element>,Unit>:
  match tree:
    case PiecesModel.Piece{+part,+first,+rows}:
      Partition.Leaf{component_out(~Element,~Context,~gather_model,components,target,target_grid,fill,context,part,first),first,rows,Unit{}}
    case PiecesModel.Pieces{left,right}:
      Partition.Fork{shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,left),
        shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,right)}

def ComponentTargets(~Element: Data,+components: Tasks.Components,+target: List<&2,U32>,+target_grid: Grid.Grid,+layout: Pieces.Ranges,+part: Models<Element>,
  +first: U32,+rows: U32) -> Data:
  TargetsAt(layout,target_grid,Tasks.target_pieces(components,target_grid,target,first))

def ResultsFacts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,tree: PiecesModel.Shaped<Models<Element>>) -> Data:
  Held.Both<{Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
        ~Tensors.component_run(~Element,~Context,~gather),Tensors.Moving{components,origins,target,target_grid,fill,context},tree) ==
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),
        shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,tree)) : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>},
    Placed.TasksAt(~Element,~Target,layout,shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,tree))>

def fork_same(~Element: Data,-left: Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,-right: Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,
  -left_to: Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,-right_to: Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,
  +left_same: {left == left_to : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>},+right_same: {right == right_to : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}) ->
  {Partition.Fork{left,right} == Partition.Fork{left_to,right_to} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}:
  %Equal.sym(Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,left,left_to,left_same) :
    {Partition.Fork{_,right} == Partition.Fork{left_to,right_to} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}
  %Equal.sym(Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,right,right_to,right_same) :
    {Partition.Fork{left_to,_} == Partition.Fork{left_to,right_to} : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}
  {==}

def results_join(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,+components: Tasks.Components,
  +origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,
  +left: PiecesModel.Shaped<Models<Element>>,+right: PiecesModel.Shaped<Models<Element>>,
  +left_facts: ResultsFacts(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,left),
  +right_facts: ResultsFacts(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,right)) ->
  ResultsFacts(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,PiecesModel.Pieces{left,right}):
  Held.Both{+left_same,+left_at} = left_facts
  Held.Both{+right_same,+right_at} = right_facts
  +moving = {Tensors.Moving{components,origins,target,target_grid,fill,context} : Tensors.Moving<Element,Context>}
  Held.Both{
    fork_same(~Element,
      Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
        ~Tensors.component_run(~Element,~Context,~gather),moving,left),
      Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
        ~Tensors.component_run(~Element,~Context,~gather),moving,right),
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),
        shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,left)),
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),
        shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,right)),
      left_same,right_same),
    Held.Both{left_at,right_at}}

# Every component of a tree whose taken pieces hold the values of their
# sources writes its target pieces with the target property.
def results_facts(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +layout: Pieces.Ranges,+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+agrees: Agrees(context,components,origins,target,target_grid,fill),
  tree: PiecesModel.Shaped<Models<Element>>,
  +held: PiecesModel.ShapedAll(~Models<Element>,~List<&2,Tensors.Origin>,~ComponentHeld(~Element,~Property,components),origins,tree),
  +targets: PiecesModel.ShapedAll(~Models<Element>,~Pieces.Ranges,~ComponentTargets(~Element,components,target,target_grid),layout,tree)) ->
  ResultsFacts(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,tree):
  match tree:
    case PiecesModel.Piece{+part,+first,+rows}:
      Held.Both{+ready,+inside} = held
      component_exact(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,components,origins,target,target_grid,fill,context,agrees,part,first,rows,ready,inside,
        targets)
    case PiecesModel.Pieces{+left,+right}:
      KernelGemm.Both{+left_held,+right_held} = held
      KernelGemm.Both{+left_targets,+right_targets} = targets
      results_join(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,left,right,
        results_facts(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,components,origins,target,target_grid,fill,context,agrees,left,left_held,left_targets),
        results_facts(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,components,origins,target,target_grid,fill,context,agrees,right,right_held,right_targets))

# ---- The components of a layout of indices ----

# The target pieces of every component of a layout of indices start leaves of
# the target layout.
def RangesTargets(ranges: Pieces.Ranges,+layout: Pieces.Ranges,+components: Tasks.Components,+target: List<&2,U32>,+target_grid: Grid.Grid) -> Data:
  match ranges:
    case Pieces.Range{+first,count}:
      TargetsAt(layout,target_grid,Tasks.target_pieces(components,target_grid,target,first))
    case Pieces.Split{left,right}:
      Held.Both<RangesTargets(left,layout,components,target,target_grid),RangesTargets(right,layout,components,target,target_grid)>

def targets_built(~Element: Data,ranges: Pieces.Ranges,+layout: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,+components: Tasks.Components,+target: List<&2,U32>,
  +target_grid: Grid.Grid,+fill: Element,+models: Models<Element>,+targets: RangesTargets(ranges,layout,components,target,target_grid)) ->
  PiecesModel.ShapedAll(~Models<Element>,~Pieces.Ranges,~ComponentTargets(~Element,components,target,target_grid),layout,
    built_inputs(~Element,built_model(~Element,ranges,origins,components,fill,models))):
  match ranges:
    case Pieces.Range{+first,+count}:
      targets
    case Pieces.Split{+left,+right}:
      Held.Both{+left_targets,+right_targets} = targets
      KernelGemm.Both{targets_built(~Element,left,layout,origins,components,target,target_grid,fill,models,left_targets),
        targets_built(~Element,right,layout,origins,components,target,target_grid,fill,built_left(~Element,built_model(~Element,left,origins,components,fill,models)),right_targets)}

# The spans of the target pieces in order onto the spans after them.
def target_spans(pieces: List<&2,U32>,+grid: Grid.Grid,+rest: Held.Spans) -> Held.Spans:
  match pieces:
    case Nil{}:
      Held.Spanned{0,0,rest}
    case Con{+piece,Nil{}}:
      Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),rest}
    case Con{+piece,more}:
      Held.Spanned{Grid.start(grid,piece),Grid.span(grid,piece),target_spans(more,grid,rest)}

def gathered_spans(~Element: Data,~Context: Data,~gather_model: GatherModel(~Element,~Context),pieces: List<&2,U32>,+context: Context,+components: Tasks.Components,
  +target_grid: Grid.Grid,+fill: Element,+holdings: Holdings<Element>,+rest: Held.Spans) ->
  {Held.layout_spans(PiecesModel.layout(~Element,pieces_of(~Element,gathered_model(~Element,~Context,~gather_model,pieces,context,components,target_grid,fill,holdings))),rest) ==
    target_spans(pieces,target_grid,rest) : Held.Spans}:
  match pieces:
    case Nil{}:
      {==}
    case Con{+piece,Nil{}}:
      {==}
    case Con{+piece,Con{+next,+more}}:
      +step = gather_model(context,components,Grid.start(target_grid,piece),Grid.span(target_grid,piece),fill,holdings)
      +after = gathered_model(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step))
      Equal.cong(Held.Spans,Held.Spans,later => Held.Spanned{Grid.start(target_grid,piece),Grid.span(target_grid,piece),later},
        Held.layout_spans(PiecesModel.layout(~Element,pieces_of(~Element,after)),rest),target_spans(Con{next,more},target_grid,rest),
        gathered_spans(~Element,~Context,~gather_model,Con{next,more},context,components,target_grid,fill,gathered_holdings(~Element,step),rest))

# The spans of the target pieces of every component of a layout of indices.
def ranges_spans(ranges: Pieces.Ranges,+components: Tasks.Components,+target: List<&2,U32>,+target_grid: Grid.Grid,+rest: Held.Spans) -> Held.Spans:
  match ranges:
    case Pieces.Range{+first,count}:
      target_spans(Tasks.target_pieces(components,target_grid,target,first),target_grid,rest)
    case Pieces.Split{left,+right}:
      ranges_spans(left,components,target,target_grid,ranges_spans(right,components,target,target_grid,rest))

def spans_built(~Element: Data,~Context: Data,~gather_model: GatherModel(~Element,~Context),ranges: Pieces.Ranges,+origins: List<&2,Tensors.Origin>,
  +components: Tasks.Components,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+models: Models<Element>,+rest: Held.Spans) ->
  {Placed.tasks_spans(~Element,shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
      built_inputs(~Element,built_model(~Element,ranges,origins,components,fill,models))),rest) ==
    ranges_spans(ranges,components,target,target_grid,rest) : Held.Spans}:
  match ranges:
    case Pieces.Range{+first,+count}:
      gathered_spans(~Element,~Context,~gather_model,Tasks.target_pieces(components,target_grid,target,first),context,components,target_grid,fill,
        held_sources(~Element,took_taken(~Element,took_model(~Element,origins,components,0,first,fill,models))),rest)
    case Pieces.Split{+left,+right}:
      +before = built_model(~Element,left,origins,components,fill,models)
      +later = built_left(~Element,before)
      +left_results = shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,built_inputs(~Element,before))
      +right_results = shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
        built_inputs(~Element,built_model(~Element,right,origins,components,fill,later)))
      Equal.trans(Held.Spans,Placed.tasks_spans(~Element,left_results,Placed.tasks_spans(~Element,right_results,rest)),
        Placed.tasks_spans(~Element,left_results,ranges_spans(right,components,target,target_grid,rest)),
        ranges_spans(left,components,target,target_grid,ranges_spans(right,components,target,target_grid,rest)),
        Equal.cong(Held.Spans,Held.Spans,after => Placed.tasks_spans(~Element,left_results,after),Placed.tasks_spans(~Element,right_results,rest),
          ranges_spans(right,components,target,target_grid,rest),spans_built(~Element,~Context,~gather_model,right,origins,components,target,target_grid,fill,context,later,rest)),
        spans_built(~Element,~Context,~gather_model,left,origins,components,target,target_grid,fill,context,models,ranges_spans(right,components,target,target_grid,rest)))

def listed_same(+layout: Pieces.Ranges,+from: Held.Spans,+to: Held.Spans,+same: {from == to : Held.Spans},+listed: Placed.LeavesListed(layout,to)) ->
  Placed.LeavesListed(layout,from):
  %Equal.sym(Held.Spans,from,to,same) : Placed.LeavesListed(layout,_)
  listed

# ---- Moved ----

def moved_exact(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,
  +parallel: Bool,+tasks: Pieces.Ranges,+count: U32,+layout: Pieces.Ranges,+models: Models<Element>,
  +results_same: {Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
        ~Tensors.component_run(~Element,~Context,~gather),Tensors.Moving{components,origins,target,target_grid,fill,context},
        built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models))) ==
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),
        shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
          built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models)))) : Partition.Tree<&1,Pieces.Pieces(~Element),Unit>}) ->
  {Tensors.moved_decided(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},Tasks.Decision{parallel,tasks},count,layout,fill,
      pack_sources(~Element,models)) ==
    PiecesModel.pack(~Element,Placed.placed_tasks(~Element,layout,fill,
      shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
        built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models))),
      Placed.empty_slots(~Element,layout,fill))) : Pieces.Pieces(~Element)}:
  +moving = {Tensors.Moving{components,origins,target,target_grid,fill,context} : Tensors.Moving<Element,Context>}
  +ranges = Tasks.component_ranges(count)
  +built = built_model(~Element,ranges,origins,components,fill,models)
  +tree = built_inputs(~Element,built)
  +results = shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,tree)
  %Equal.sym(Tensors.Sources<Element> & Partition.Tree<&1,Tensors.Sources<Element>,Unit>,
      Pieces.build(~Tensors.Sources<Element>,~Tensors.Sources<Element>,~Tensors.Moving<Element,Context>,~Tensors.component_inputs(~Element,~Context),ranges,moving,
        pack_sources(~Element,models)),
      built_pair(~Element,built),built_exact(~Element,~Context,ranges,origins,components,target,target_grid,fill,context,models)) :
    {Tensors.placed(~Element,layout,fill,Pieces.grouped(~Tensors.Sources<Element>,~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
        ~Tensors.component_run(~Element,~Context,~gather),tasks,parallel,moving,
        Pair.snd(Tensors.Sources<Element>,Partition.Tree<&1,Tensors.Sources<Element>,Unit>,_))) ==
      PiecesModel.pack(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))) : Pieces.Pieces(~Element)}
  %Equal.sym(Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,
      Pieces.grouped(~Tensors.Sources<Element>,~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,~Tensors.component_run(~Element,~Context,~gather),tasks,parallel,moving,
        PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),tree)),
      PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results),
      Equal.trans(Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,
        Pieces.grouped(~Tensors.Sources<Element>,~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,~Tensors.component_run(~Element,~Context,~gather),tasks,parallel,moving,
          PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),tree)),
        Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
          ~Tensors.component_run(~Element,~Context,~gather),moving,tree),
        PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results),
        Equal.trans(Partition.Tree<&1,Pieces.Pieces(~Element),Unit>,
          Pieces.grouped(~Tensors.Sources<Element>,~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,~Tensors.component_run(~Element,~Context,~gather),tasks,parallel,moving,
            PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),tree)),
          Pieces.group_run(~Tensors.Sources<Element>,~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,~Tensors.component_run(~Element,~Context,~gather),moving,
            PiecesModel.shaped_pack(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),tree)),
          Each.shaped_leaf_map(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
            ~Tensors.component_run(~Element,~Context,~gather),moving,tree),
          Walk.grouped(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
            ~Tensors.component_run(~Element,~Context,~gather),tasks,parallel,moving,tree),
          Each.shaped_group_serial(~Models<Element>,~Tensors.Sources<Element>,~pack_sources(~Element),~Pieces.Pieces(~Element),~Tensors.Moving<Element,Context>,
            ~Tensors.component_run(~Element,~Context,~gather),moving,tree)),
        results_same)) :
    {Tensors.placed(~Element,layout,fill,_) == PiecesModel.pack(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))) : Pieces.Pieces(~Element)}
  Placed.placed_exact(~Element,layout,fill,results)

def moved_from(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,
  +parallel: Bool,+tasks: Pieces.Ranges,+count: U32,+layout: Pieces.Ranges,+models: Models<Element>,
  +facts: ResultsFacts(~Element,~Context,~gather,~gather_model,~Target,layout,components,origins,target,target_grid,fill,context,
    built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models))),
  +ordered: Placed.Ordered(layout),
  +listed: Placed.LeavesListed(layout,Placed.tasks_spans(~Element,shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
    built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models))),Held.Ended{}))) ->
  Placed.PlacedModel<Element,Target,layout,Tensors.moved_decided(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},
    Tasks.Decision{parallel,tasks},count,layout,fill,pack_sources(~Element,models))>:
  Held.Both{+same,+tasks_at} = facts
  +results = shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,
    built_inputs(~Element,built_model(~Element,Tasks.component_ranges(count),origins,components,fill,models)))
  +slots = Placed.empty_slots(~Element,layout,fill)
  Placed.PlacedModel{Placed.placed_tasks(~Element,layout,fill,results,slots),
    moved_exact(~Element,~Context,~gather,~gather_model,components,origins,target,target_grid,fill,context,parallel,tasks,count,layout,models,same),
    Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,slots)),PiecesModel.layout(~Element,slots),layout,
      Placed.placed_tasks_layout(~Element,layout,fill,results,slots),Placed.slots_layout(~Element,layout,fill)),
    Placed.placed_listed(~Element,~Target,layout,fill,results,ordered,tasks_at,listed)}

# Tensors.moved_decided: when the sources are laid in order with their
# values, every component lists source pieces no earlier one took, its target
# pieces start leaves of the target layout and together reach every leaf,
# and every gather holds its law, every target piece has the target property.
def kept_model(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~Kept: @+grid: Grid.Grid -> @+models: Models<Element> -> @+holdings: Holdings<Element> -> @+pieces: List<&2,U32> -> Data,
  ~gathers: KeptLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept),~begin: KeptBegin(~Element,~Context,~Property,~Agrees,~Kept),
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+agrees: Agrees(context,components,origins,target,target_grid,fill),
  +parallel: Bool,+tasks: Pieces.Ranges,+count: U32,+layout: Pieces.Ranges,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,0,fill,models,gones),+good: SourcesGood(~Element,~Property,origins,0,models,gones),
  +ready: BuildReady(Tasks.component_ranges(count),origins,components,gones),+targets: RangesTargets(Tasks.component_ranges(count),layout,components,target,target_grid),
  +ordered: Placed.Ordered(layout),+listed: Placed.LeavesListed(layout,ranges_spans(Tasks.component_ranges(count),components,target,target_grid,Held.Ended{}))) ->
  Placed.PlacedModel<Element,Target,layout,Tensors.moved_decided(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},
    Tasks.Decision{parallel,tasks},count,layout,fill,pack_sources(~Element,models))>:
  +ranges = Tasks.component_ranges(count)
  +tree = built_inputs(~Element,built_model(~Element,ranges,origins,components,fill,models))
  +results = shaped_results(~Element,~Context,~gather_model,components,target,target_grid,fill,context,tree)
  +slots = Placed.empty_slots(~Element,layout,fill)
  +facts = results_facts(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~Kept,~gathers,~begin,layout,components,origins,target,target_grid,fill,context,agrees,tree,
    held_built(~Element,~Property,ranges,origins,components,fill,models,gones,static,ready,good),
    targets_built(~Element,ranges,layout,origins,components,target,target_grid,fill,models,targets))
  moved_from(~Element,~Context,~gather,~gather_model,~Target,components,origins,target,target_grid,fill,context,parallel,tasks,count,layout,models,facts,ordered,
    listed_same(layout,Placed.tasks_spans(~Element,results,Held.Ended{}),ranges_spans(ranges,components,target,target_grid,Held.Ended{}),
      spans_built(~Element,~Context,~gather_model,ranges,origins,components,target,target_grid,fill,context,models,Held.Ended{}),listed))


# Tensors.moved_decided, with gathers that keep the holdings in order: when the sources are laid in order with their
# values, every component lists source pieces no earlier one took, its target
# pieces start leaves of the target layout and together reach every leaf,
# and every gather holds its law, every target piece has the target property.
def moved_model(~Element: Data,~Context: Data,~gather: Tensors.Gather(~Element,~Context),~gather_model: GatherModel(~Element,~Context),
  ~Property: @+source: U32 -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Target: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Agrees: @+context: Context -> @+components: Tasks.Components -> @+origins: List<&2,Tensors.Origin> -> @+target: List<&2,U32> -> @+target_grid: Grid.Grid -> @+fill: Element -> Data,
  ~gathers: GatherLaw(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees),
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+target: List<&2,U32>,+target_grid: Grid.Grid,+fill: Element,+context: Context,+agrees: Agrees(context,components,origins,target,target_grid,fill),
  +parallel: Bool,+tasks: Pieces.Ranges,+count: U32,+layout: Pieces.Ranges,+models: Models<Element>,+gones: Gones,
  +static: SourcesStatic(~Element,~Property,origins,0,fill,models,gones),+good: SourcesGood(~Element,~Property,origins,0,models,gones),
  +ready: BuildReady(Tasks.component_ranges(count),origins,components,gones),+targets: RangesTargets(Tasks.component_ranges(count),layout,components,target,target_grid),
  +ordered: Placed.Ordered(layout),+listed: Placed.LeavesListed(layout,ranges_spans(Tasks.component_ranges(count),components,target,target_grid,Held.Ended{}))) ->
  Placed.PlacedModel<Element,Target,layout,Tensors.moved_decided(~Element,~Context,~gather,Tensors.Moving{components,origins,target,target_grid,fill,context},
    Tasks.Decision{parallel,tasks},count,layout,fill,pack_sources(~Element,models))>:
  kept_model(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,~ordered_kept(~Element),
    ~ordered_law(~Element,~Context,~gather,~gather_model,~Property,~Target,~Agrees,gathers),~ordered_begin(~Element,~Context,~Property,~Agrees),
    components,origins,target,target_grid,fill,context,agrees,parallel,tasks,count,layout,models,gones,static,good,ready,targets,ordered,listed)