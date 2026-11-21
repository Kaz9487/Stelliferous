# The gather of placed sources. A target piece starts as fill and takes from
# every source in turn the parts of the piece that the source places, so a
# cell ends with the source cell of the last source whose segments cover it.
import Base
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_move.bend as Move
import ../tensor_tasks.bend as Tasks
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./tensor_move_parts.bend as MoveParts
import ./tensor_moved.bend as Moved
import ./tensor_placements.bend as Placements
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_grid_box.bend as Box
import ./tensor_box_digits.bend as Digits
import ./tensor_pieces_content.bend as Content
import ./tensor_placed.bend as Placed
import ./tensor_view_gather.bend as ViewGather
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_allocation_depth.bend as Depth
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_readback.bend as Readback
import ./word_power_of_two.bend as Power

# ---- The model ----

# The holdings and the target piece after every source places its parts.
def placed_model(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,holdings: Moved.Holdings<Element>,+output: Buffer.Model<Element>) -> Moved.Gathered<Element>:
  match segments sources holdings:
    case Con{+own,segments} Con{+source,sources} Moved.Holds{+holding,rest}:
      +list = Tasks.placements(own,source,grid,cells,first,count)
      +later = placed_model(~Element,segments,sources,grid,cells,first,count,rest,
        Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)))))
      Moved.Gathered{Moved.Holds{Held.ran_holding(~Element,MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output))),Moved.gathered_holdings(~Element,later)},
        Moved.gathered_output(~Element,later)}
    case segments sources holdings: Moved.Gathered{holdings,output}

# The target tree after every source places its parts.
def placed_tree(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,holdings: Moved.Holdings<Element>,+tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match segments sources holdings:
    case Con{+own,segments} Con{+source,sources} Moved.Holds{+holding,rest}:
      placed_tree(~Element,segments,sources,grid,cells,first,count,rest,
        Held.ran_tree(~Element,MoveParts.parts_ran(~Element,Tasks.placements(own,source,grid,cells,first,count),holding,tree)))
    case segments sources holdings: tree

# The model of a gather of placed sources.
def placing_model(~Element: Data,+context: Tensors.Placing,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Moved.Holdings<Element>) ->
  Moved.Gathered<Element>:
  Tensors.Placing{+target_grid,+target,+sources,+segments} = context
  placed_model(~Element,segments,sources,target_grid,target,first,count,holdings,ViewGather.fresh_output(~Element,fill,count))

# ---- The gather is its model ----

# One source placed: its holding waits while the later sources are placed,
# then joins their holdings.
def sources_step(~Element: Data,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,+rest: Moved.Holdings<Element>,-waiting: Tensors.Waiting<Element>,pair: Move.Holding<Element> & Storage.Buffer<Element>,+held: Held.Holding<Element>,
  +after: Buffer.Model<Element>,+later: Moved.Gathered<Element>,
  +held_eq: {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,pair) == Held.pack(~Element,held) : Move.Holding<Element>},
  +buffer_eq: {Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,pair) == Buffer.pack(~Element,after) : Storage.Buffer<Element>},
  +rest_eq: {Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,rest),
      Tensors.Placed{Tensors.Unheld{},Tensors.Waits{Held.pack(~Element,held),waiting},Buffer.pack(~Element,after)}) ==
    Tensors.Placed{Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later)),Tensors.Waits{Held.pack(~Element,held),waiting},
      Buffer.pack(~Element,Moved.gathered_output(~Element,later))} : Tensors.Placed<Element>}) ->
  {Tensors.placed_joined(~Element,Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,rest),
      Tensors.placed_after(~Element,waiting,pair))) ==
    Tensors.Placed{Tensors.Holds{Held.pack(~Element,held),Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later))},waiting,
      Buffer.pack(~Element,Moved.gathered_output(~Element,later))} : Tensors.Placed<Element>}:
  (got,written) = pair
  %Equal.sym(Move.Holding<Element>,got,Held.pack(~Element,held),held_eq) :
    {Tensors.placed_joined(~Element,Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,rest),
        Tensors.Placed{Tensors.Unheld{},Tensors.Waits{_,waiting},written})) ==
      Tensors.Placed{Tensors.Holds{Held.pack(~Element,held),Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later))},waiting,
        Buffer.pack(~Element,Moved.gathered_output(~Element,later))} : Tensors.Placed<Element>}
  %Equal.sym(Storage.Buffer<Element>,written,Buffer.pack(~Element,after),buffer_eq) :
    {Tensors.placed_joined(~Element,Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,rest),
        Tensors.Placed{Tensors.Unheld{},Tensors.Waits{Held.pack(~Element,held),waiting},_})) ==
      Tensors.Placed{Tensors.Holds{Held.pack(~Element,held),Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later))},waiting,
        Buffer.pack(~Element,Moved.gathered_output(~Element,later))} : Tensors.Placed<Element>}
  %Equal.sym(Tensors.Placed<Element>,Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,rest),
        Tensors.Placed{Tensors.Unheld{},Tensors.Waits{Held.pack(~Element,held),waiting},Buffer.pack(~Element,after)}),
      Tensors.Placed{Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later)),Tensors.Waits{Held.pack(~Element,held),waiting},
        Buffer.pack(~Element,Moved.gathered_output(~Element,later))},rest_eq) :
    {Tensors.placed_joined(~Element,_) ==
      Tensors.Placed{Tensors.Holds{Held.pack(~Element,held),Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,later))},waiting,
        Buffer.pack(~Element,Moved.gathered_output(~Element,later))} : Tensors.Placed<Element>}
  {==}

# Placing sources from holdings into a buffer is the model; what waits stays.
def placed_written(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,holdings: Moved.Holdings<Element>,-done: Tensors.Holdings<Element>,-waiting: Tensors.Waiting<Element>,+output: Buffer.Model<Element>) ->
  {Tensors.placed_sources(~Element,segments,sources,grid,cells,first,count,Moved.pack_holdings(~Element,holdings),Tensors.Placed{done,waiting,Buffer.pack(~Element,output)}) ==
    Tensors.Placed{Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,placed_model(~Element,segments,sources,grid,cells,first,count,holdings,output))),waiting,
      Buffer.pack(~Element,Moved.gathered_output(~Element,placed_model(~Element,segments,sources,grid,cells,first,count,holdings,output)))} : Tensors.Placed<Element>}:
  match segments:
    case Nil{}: {==}
    case Con{+own,+segments}:
      match sources:
        case Nil{}: {==}
        case Con{+source,+sources}:
          match holdings:
            case Moved.Unheld{}: {==}
            case Moved.Holds{+holding,+rest}:
              +list = Tasks.placements(own,source,grid,cells,first,count)
              +after = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output))))
              +ran = MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output))
              sources_step(~Element,segments,sources,grid,cells,first,count,rest,waiting,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)),
                Held.ran_holding(~Element,ran),after,placed_model(~Element,segments,sources,grid,cells,first,count,rest,after),
                Held.run_held(~Element,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)),ran,Buffer.length(~Element,output),MoveParts.parts_refines(~Element,list,holding,output)),
                Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,after),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output))),
                  Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output))))),
                placed_written(~Element,segments,sources,grid,cells,first,count,rest,Tensors.Unheld{},Tensors.Waits{Held.pack(~Element,Held.ran_holding(~Element,ran)),waiting},after))

# A gather of placed sources is its model.
def placing_written(~Element: Data,+context: Tensors.Placing,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Moved.Holdings<Element>) ->
  {Tensors.placing_gather(~Element,context,components,first,count,fill,Moved.pack_holdings(~Element,holdings)) ==
    (Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,placing_model(~Element,context,components,first,count,fill,holdings))),
      Buffer.pack(~Element,Moved.gathered_output(~Element,placing_model(~Element,context,components,first,count,fill,holdings)))) :
    Tensors.Holdings<Element> & Storage.Buffer<Element>}:
  Tensors.Placing{+target_grid,+target,+sources,+segments} = context
  +output = ViewGather.fresh_output(~Element,fill,count)
  +model = placed_model(~Element,segments,sources,target_grid,target,first,count,holdings,output)
  %Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count)) :
    {Tensors.placed_result(~Element,Tensors.placed_sources(~Element,segments,sources,target_grid,target,first,count,Moved.pack_holdings(~Element,holdings),
        Tensors.Placed{Tensors.Unheld{},Tensors.NoneWaiting{},_})) ==
      (Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,model)),Buffer.pack(~Element,Moved.gathered_output(~Element,model))) :
      Tensors.Holdings<Element> & Storage.Buffer<Element>}
  Equal.cong(Tensors.Placed<Element>,Tensors.Holdings<Element> & Storage.Buffer<Element>,placed => Tensors.placed_result(~Element,placed),
    Tensors.placed_sources(~Element,segments,sources,target_grid,target,first,count,Moved.pack_holdings(~Element,holdings),
      Tensors.Placed{Tensors.Unheld{},Tensors.NoneWaiting{},Buffer.pack(~Element,output)}),
    Tensors.Placed{Moved.pack_holdings(~Element,Moved.gathered_holdings(~Element,model)),Tensors.NoneWaiting{},Buffer.pack(~Element,Moved.gathered_output(~Element,model))},
    placed_written(~Element,segments,sources,target_grid,target,first,count,holdings,Tensors.Unheld{},Tensors.NoneWaiting{},output))

# ---- The holdings and the target ----

# Placing keeps every holding in order.
def placed_order(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,holdings: Moved.Holdings<Element>,models: Moved.Models<Element>,+output: Buffer.Model<Element>,+order: Moved.HeldOrder(~Element,holdings,models)) ->
  Moved.HeldOrder(~Element,Moved.gathered_holdings(~Element,placed_model(~Element,segments,sources,grid,cells,first,count,holdings,output)),models):
  match segments:
    case Nil{}: order
    case Con{+own,+segments}:
      match sources:
        case Nil{}: order
        case Con{+source,+sources}:
          match holdings models:
            case Moved.Holds{+holding,+rest} Moved.Models{+model,+more}:
              Held.Both{+ordered,+later} = order
              +list = Tasks.placements(own,source,grid,cells,first,count)
              +after = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output))))
              Held.Both{Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output)))),
                  Held.order(~Element,holding),Held.stacked_model(~Element,model,Held.Bottom{}),MoveParts.parts_order(~Element,list,holding,Buffer.tree(~Element,output)),ordered),
                placed_order(~Element,segments,sources,grid,cells,first,count,rest,more,after,later)}
            case Moved.Unheld{} Moved.NoModels{}: Unit{}
            case Moved.Holds{+holding,+rest} Moved.NoModels{}:
              Empty.absurd(Moved.HeldOrder(~Element,Moved.gathered_holdings(~Element,placed_model(~Element,Con{own,segments},Con{source,sources},grid,cells,first,count,
                Moved.Holds{holding,rest},output)),Moved.NoModels{}),order)
            case Moved.Unheld{} Moved.Models{+model,+more}: Empty.absurd(Moved.HeldOrder(~Element,Moved.Unheld{},Moved.Models{model,more}),order)

# The target after placing holds the placed tree, as long as before.
def placed_content(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,
  +count: U32,holdings: Moved.Holdings<Element>,+output: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,
  +exact: {Buffer.content(~Element,output) == Buffer.Content{tree,length} : Buffer.Content<Element>}) ->
  {Buffer.content(~Element,Moved.gathered_output(~Element,placed_model(~Element,segments,sources,grid,cells,first,count,holdings,output))) ==
    Buffer.Content{placed_tree(~Element,segments,sources,grid,cells,first,count,holdings,tree),length} : Buffer.Content<Element>}:
  match segments:
    case Nil{}: exact
    case Con{+own,+segments}:
      match sources:
        case Nil{}: exact
        case Con{+source,+sources}:
          match holdings:
            case Moved.Unheld{}: exact
            case Moved.Holds{+holding,+rest}:
              +list = Tasks.placements(own,source,grid,cells,first,count)
              +after = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output))))
              +same_tree = Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,output),Buffer.content_tree(~Element,Buffer.content(~Element,output)),tree,
                Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,output)),Buffer.tree(~Element,output),Buffer.content_tree_of(~Element,output)),
                Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,output),Buffer.Content{tree,length},exact))
              +same_length = Equal.trans(Nat,Buffer.length(~Element,output),Buffer.content_length(~Element,Buffer.content(~Element,output)),length,
                Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,output)),Buffer.length(~Element,output),Buffer.content_length_of(~Element,output)),
                Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,output),Buffer.Content{tree,length},exact))
              +after_content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,after),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)))),
                Buffer.Content{Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output))),Buffer.length(~Element,output)},
                Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)))),
                Held.run_target(~Element,Move.parts(~Element,list,Held.pack(~Element,holding),Buffer.pack(~Element,output)),MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output)),Buffer.length(~Element,output),
                  MoveParts.parts_refines(~Element,list,holding,output)))
              placed_content(~Element,segments,sources,grid,cells,first,count,rest,after,Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,tree)),length,
                {%same_tree : {Buffer.content(~Element,after) == Buffer.Content{Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,_)),length} : Buffer.Content<Element>}
                  %same_length : {Buffer.content(~Element,after) == Buffer.Content{Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,Buffer.tree(~Element,output))),_} :
                    Buffer.Content<Element>}
                  after_content :
                  {Buffer.content(~Element,after) == Buffer.Content{Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,tree)),length} : Buffer.Content<Element>}})

# ---- The value of a cell of the target ----

# The value the sources give a cell at coordinates, from old: a source whose
# segments cover the coordinates on every axis gives the value of the source
# cell they read.
def sources_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logicals: List<&2,Logical>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+coordinates: List<&2,U32>,+old: Element) -> Element:
  match logicals segments sources:
    case Con{+logical,logicals} Con{+own,segments} Con{+source,sources}:
      sources_value(~Element,~Logical,~original,logicals,segments,sources,coordinates,
        Placements.placed_result(~Element,~Logical,~original,logical,Placements.segments_cell(own,coordinates,Tasks.strides_of(source)),old))
    case logicals segments sources: old

# What one source needs to place its parts: its holding in order over its
# cells and holding its values, the parts ready, and its axes in the piece.
def SourceReady(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+own: List<&2,List<&2,Tasks.Segment>>,+source: List<&2,U32>,+holding: Held.Holding<Element>,+grid: Grid.Grid,
  +cells: List<&2,U32>,+first: U32,+count: U32,+cap: U32,+word: U32,+capacity: U32) -> Data:
  Held.Both<{Nat.is_le(ShapeLaws.elements(source),U32.to_nat(cap)) == True{} : Bool},
    Held.Both<Held.Sorted(~Element,Held.order(~Element,holding),0n,ShapeLaws.elements(source)),
      Held.Both<Held.HoldingValues(~Element,~Logical,~original,logical,holding),
        Held.Both<MoveParts.PartsReady(Tasks.placements(own,source,grid,cells,first,count),cap,word,Held.spans(~Element,Held.order(~Element,holding)),capacity),
          Placements.AxesIn(own,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),cap)>>>>

def PlacedReady(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logicals: List<&2,Logical>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,holdings: Moved.Holdings<Element>,+grid: Grid.Grid,+cells: List<&2,U32>,
  +first: U32,+count: U32,+cap: U32,+word: U32,+capacity: U32) -> Data:
  match logicals segments sources holdings:
    case Con{+logical,logicals} Con{+own,segments} Con{+source,sources} Moved.Holds{+holding,rest}:
      Held.Both<SourceReady(~Element,~Logical,~original,logical,own,source,holding,grid,cells,first,count,cap,word,capacity),
        PlacedReady(~Element,~Logical,~original,logicals,segments,sources,rest,grid,cells,first,count,cap,word,capacity)>
    case Nil{} Nil{} Nil{} holdings: Unit
    case logicals segments sources holdings: Empty

# Source after source, a cell of the piece goes from its value in the tree to
# the value the sources give its coordinates.
def placed_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logicals: List<&2,Logical>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,holdings: Moved.Holdings<Element>,+grid: Grid.Grid,+cells: List<&2,U32>,
  +first: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+capacity: U32,+capacity_eq: {Tree.capacity(Element,tree) == capacity : U32},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Grid.box_of(grid,cells,count))),U32.to_nat(cap)) == True{} : Bool},+query: U32,
  +query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(Grid.box_of(grid,cells,count))) == True{} : Bool},+query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+old: Element,
  +start: {Tree.read(Element,tree,query) == old : Element},+ready: PlacedReady(~Element,~Logical,~original,logicals,segments,sources,holdings,grid,cells,first,count,cap,word,capacity)) ->
  {Tree.read(Element,placed_tree(~Element,segments,sources,grid,cells,first,count,holdings,tree),query) == sources_value(~Element,~Logical,~original,logicals,segments,sources,Placements.positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old) : Element}:
  match logicals:
    case Nil{}:
      match segments:
        case Nil{}: start
        case Con{+own,+more}:
          match sources:
            case Nil{}: start
            case Con{+source,+others}:
              match holdings:
                case Moved.Unheld{}: start
                case Moved.Holds{+holding,+rest}:
                  Empty.absurd({Tree.read(Element,placed_tree(~Element,Con{own,more},Con{source,others},grid,cells,first,count,Moved.Holds{holding,rest},tree),query) == old : Element},ready)
    case Con{+logical,+logicals}:
      match segments:
        case Nil{}: start
        case Con{+own,+segments}:
          match sources:
            case Nil{}: start
            case Con{+source,+sources}:
              match holdings:
                case Moved.Unheld{}:
                  Empty.absurd({Tree.read(Element,placed_tree(~Element,Con{own,segments},Con{source,sources},grid,cells,first,count,Moved.Unheld{},tree),query) ==
                    sources_value(~Element,~Logical,~original,Con{logical,logicals},Con{own,segments},Con{source,sources},Placements.positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old) : Element},ready)
                case Moved.Holds{+holding,+rest}:
                  Held.Both{Held.Both{+bounded,Held.Both{+sorted,Held.Both{+values,Held.Both{+parts_ready,+axes}}}},+later} = ready
                  +ran = MoveParts.parts_ran(~Element,Tasks.placements(own,source,grid,cells,first,count),holding,tree)
                  +result = Placements.placed_result(~Element,~Logical,~original,logical,Placements.segments_cell(own,Placements.positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old)
                  +at = Placements.placements_at(~Element,~Logical,~original,logical,own,source,grid,cells,first,count,cap,cap_ok,piece_room,query,query_in,old,axes)
                  +cell = MoveParts.parts_cell(~Element,~Logical,~original,logical,Tasks.placements(own,source,grid,cells,first,count),Placements.prefix_indices(Placements.placed_prefixes(own,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Placements.Prefix{[],0,0}]),query),holding,tree,depth,balanced,depth_ok,
                    {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_eq) : {U32.is_le(_,2147483648) == True{} : Bool} capacity_ok :
                      {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool}},
                    cap,cap_ok,word,word_ok,Held.order(~Element,holding),{==},0n,ShapeLaws.elements(source),cap,cap_ok,bounded,sorted,values,
                    {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_eq) : MoveParts.PartsReady(Tasks.placements(own,source,grid,cells,first,count),cap,word,Held.spans(~Element,Held.order(~Element,holding)),_)
                      parts_ready : MoveParts.PartsReady(Tasks.placements(own,source,grid,cells,first,count),cap,word,Held.spans(~Element,Held.order(~Element,holding)),Tree.capacity(Element,tree))},
                    query,{%Equal.sym(U32,Tree.capacity(Element,tree),capacity,capacity_eq) : {U32.is_lt(query,_) == True{} : Bool} query_inside :
                      {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}},result,
                    {%Equal.sym(Element,Tree.read(Element,tree,query),old,start) : MoveParts.PartsAt(~Element,~Logical,~original,logical,Tasks.placements(own,source,grid,cells,first,count),Placements.prefix_indices(Placements.placed_prefixes(own,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Placements.Prefix{[],0,0}]),query),query,_,result)
                      at : MoveParts.PartsAt(~Element,~Logical,~original,logical,Tasks.placements(own,source,grid,cells,first,count),Placements.prefix_indices(Placements.placed_prefixes(own,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Placements.Prefix{[],0,0}]),query),query,Tree.read(Element,tree,query),result)})
                  placed_cell(~Element,~Logical,~original,logicals,segments,sources,rest,grid,cells,first,count,cap,cap_ok,word,word_ok,Held.ran_tree(~Element,ran),depth,
                    MoveParts.parts_balanced(~Element,Tasks.placements(own,source,grid,cells,first,count),holding,tree,depth,balanced),depth_ok,capacity,
                    Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),capacity,MoveParts.parts_capacity(~Element,Tasks.placements(own,source,grid,cells,first,count),holding,tree),capacity_eq),
                    capacity_ok,piece_room,query,query_in,query_inside,result,cell,later)

# ---- The coordinates of a cell of a piece ----

# Every number at most cap.
def NumbersIn(numbers: List<&2,Nat>,+cap: U32) -> Data:
  match numbers:
    case Nil{}: Unit
    case Con{+number,numbers}: Held.Both<{Nat.is_le(number,U32.to_nat(cap)) == True{} : Bool},NumbersIn(numbers,cap)>

# The coordinates of a cell are at most the cell.
def coordinates_in(extents: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool},+cap: U32,
  +room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) -> NumbersIn(Targets.cell_coordinates(extents,index),cap):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      +inner = ShapeLaws.elements(rest)
      +positive = Digits.elements_positive(extent,rest,index,inside)
      +quotient = Nat.div(index,inner)
      +remainder = Nat.mod(index,inner)
      Held.Both{Order.transitive(quotient,index,U32.to_nat(cap),
          Order.transitive(quotient,Nat.mul(quotient,inner),index,RowMajor.product_covers(quotient,inner,positive),DivisionOrder.product_below(index,inner,positive)),room),
        coordinates_in(rest,remainder,DivisionOrder.remainder_small(index,inner,positive),cap,
          Order.transitive(remainder,index,U32.to_nat(cap),DivisionOrder.remainder_le(index,inner,positive),room))}

# Coordinates plus digits that stay inside cap number the sums.
def positions_words(coordinates: List<&2,U32>,digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +within: NumbersIn(Digits.added(Digits.words_nat(coordinates),digits),cap)) ->
  {Digits.words_nat(Placements.positions(coordinates,digits)) == Digits.added(Digits.words_nat(coordinates),digits) : List<&2,Nat>}:
  match coordinates:
    case Nil{}: {==}
    case Con{+low,+rest}:
      match digits:
        case Nil{}: {==}
        case Con{+digit,+more}:
          Held.Both{+head_in,+rest_in} = within
          +sum = Nat.add(U32.to_nat(low),digit)
          +digit_room = Order.transitive(digit,sum,U32.to_nat(cap),
            {%Order.commutative(digit,U32.to_nat(low)) : {Nat.is_le(digit,_) == True{} : Bool} Order.le_add(digit,U32.to_nat(low)) : {Nat.is_le(digit,sum) == True{} : Bool}},head_in)
          +head = Arithmetic.sum_exact(cap,cap_ok,low,U32.from_nat(digit),U32.to_nat(low),digit,{==},Bounds.roundtrip(digit,cap,cap_ok,digit_room),head_in)
          +tail = positions_words(rest,more,cap,cap_ok,rest_in)
          %head : {Con{U32.to_nat((low + U32.from_nat(digit) : U32)),Digits.words_nat(Placements.positions(rest,more))} ==
            Con{_,Digits.added(Digits.words_nat(rest),more)} : List<&2,Nat>}
          %tail : {Con{U32.to_nat((low + U32.from_nat(digit) : U32)),Digits.words_nat(Placements.positions(rest,more))} ==
            Con{U32.to_nat((low + U32.from_nat(digit) : U32)),_} : List<&2,Nat>}
          {==}

def after_first(numbers: List<&2,Nat>) -> List<&2,Nat>:
  match numbers:
    case Nil{}: []
    case Con{+number,+rest}: rest

# Words with the same numbers are the same.
def words_same(left: List<&2,U32>,right: List<&2,U32>,+same: {Digits.words_nat(left) == Digits.words_nat(right) : List<&2,Nat>}) -> {left == right : List<&2,U32>}:
  match left right:
    case Nil{} Nil{}: {==}
    case Con{+word,+more} Con{+other,+rest}:
      +head = Arithmetic.to_nat_injective(word,other,
        Equal.cong(List<&2,Nat>,Nat,numbers => Digits.digit_at(numbers,0n),Digits.words_nat(Con{word,more}),Digits.words_nat(Con{other,rest}),same))
      +tail = words_same(more,rest,Equal.cong(List<&2,Nat>,List<&2,Nat>,numbers => after_first(numbers),Digits.words_nat(Con{word,more}),Digits.words_nat(Con{other,rest}),same))
      %head : {Con{word,more} == Con{_,rest} : List<&2,U32>}
      %tail : {Con{word,more} == Con{word,_} : List<&2,U32>}
      {==}
    case Nil{} Con{+other,+rest}:
      Empty.absurd({Nil{} == Con{other,rest} : List<&2,U32>},Digits.zero_successor(List.length(&2,Nat,Digits.words_nat(rest)),
        Equal.cong(List<&2,Nat>,Nat,numbers => List.length(&2,Nat,numbers),Digits.words_nat(Nil{}),Digits.words_nat(Con{other,rest}),same)))
    case Con{+word,+more} Nil{}:
      Empty.absurd({Con{word,more} == Nil{} : List<&2,U32>},Digits.zero_successor(List.length(&2,Nat,Digits.words_nat(more)),
        Equal.cong(List<&2,Nat>,Nat,numbers => List.length(&2,Nat,numbers),Digits.words_nat(Nil{}),Digits.words_nat(Con{word,more}),
          Equal.sym(List<&2,Nat>,Digits.words_nat(Con{word,more}),Digits.words_nat(Nil{}),same))))

# The target coordinates of a cell of a piece are the coordinates of its
# address.
def piece_positions(+extents: List<&2,U32>,+boxes: List<&2,U32>,+first: U32,+cells: Nat,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap)) == True{} : Bool},
  +same: {List.length(&2,U32,boxes) == List.length(&2,U32,extents) : Nat},
  +ending: {Nat.is_lt(Nat.add(U32.to_nat(first),index),ShapeLaws.elements(extents)) == True{} : Bool},
  +boxed: Digits.Boxes(List.length(&2,U32,extents),extents,boxes,U32.to_nat(first),cells,index,0n)) ->
  {Placements.positions(Grid.coordinates(extents,first),Targets.cell_coordinates(boxes,index)) == Grid.coordinates(extents,U32.from_nat(Nat.add(U32.to_nat(first),index))) : List<&2,U32>}:
  +start = U32.to_nat(first)
  +cell = Nat.add(start,index)
  +elements = ShapeLaws.elements(extents)
  +digits = Targets.cell_coordinates(boxes,index)
  +corner = Grid.coordinates(extents,first)
  +single = Order.transitive(elements,Nat.double(elements),U32.to_nat(cap),Order.le_double(elements),room)
  +cell_room = Order.transitive(cell,elements,U32.to_nat(cap),Bounds.strict_le(cell,elements,ending),single)
  +back = Bounds.roundtrip(cell,cap,cap_ok,cell_room)
  +lows = Digits.coordinates_exact(extents,first,cap,cap_ok,room,Order.le_lt_trans(start,cell,elements,Order.le_add(start,index),ending))
  +sums = Equal.trans(List<&2,Nat>,Digits.added(Digits.words_nat(corner),digits),Digits.added(Targets.cell_coordinates(extents,start),digits),Targets.cell_coordinates(extents,cell),
    Equal.cong(List<&2,Nat>,List<&2,Nat>,numbers => Digits.added(numbers,digits),Digits.words_nat(corner),Targets.cell_coordinates(extents,start),lows),
    Digits.piece_coordinates(extents,boxes,start,cells,index,cap,cap_ok,single,same,boxed))
  +within = {%Equal.sym(List<&2,Nat>,Digits.added(Digits.words_nat(corner),digits),Targets.cell_coordinates(extents,cell),sums) : NumbersIn(_,cap)
    coordinates_in(extents,cell,ending,cap,cell_room) : NumbersIn(Digits.added(Digits.words_nat(corner),digits),cap)}
  +target = Digits.coordinates_exact(extents,U32.from_nat(cell),cap,cap_ok,room,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(cell)),cell,back) : {Nat.is_lt(_,elements) == True{} : Bool} ending : {Nat.is_lt(U32.to_nat(U32.from_nat(cell)),elements) == True{} : Bool}})
  words_same(Placements.positions(corner,digits),Grid.coordinates(extents,U32.from_nat(cell)),
    Equal.trans(List<&2,Nat>,Digits.words_nat(Placements.positions(corner,digits)),Digits.added(Digits.words_nat(corner),digits),Digits.words_nat(Grid.coordinates(extents,U32.from_nat(cell))),
      positions_words(corner,digits,cap,cap_ok,within),
      Equal.trans(List<&2,Nat>,Digits.added(Digits.words_nat(corner),digits),Targets.cell_coordinates(extents,cell),Digits.words_nat(Grid.coordinates(extents,U32.from_nat(cell))),sums,
        Equal.sym(List<&2,Nat>,Digits.words_nat(Grid.coordinates(extents,U32.from_nat(cell))),Targets.cell_coordinates(extents,cell),
          Equal.trans(List<&2,Nat>,Digits.words_nat(Grid.coordinates(extents,U32.from_nat(cell))),Targets.cell_coordinates(extents,U32.to_nat(U32.from_nat(cell))),
            Targets.cell_coordinates(extents,cell),target,
            Equal.cong(Nat,List<&2,Nat>,number => Targets.cell_coordinates(extents,number),U32.to_nat(U32.from_nat(cell)),cell,back))))))

# ---- What a placing writes into a target piece ----

# The sources placed in a target: the logical of every source, the segments
# and the cells of every source, the target cells and the fill.
type Placings<-Logical: Data,-Element: Data> is Data:
  Placings{logicals: List<&2,Logical>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,target: List<&2,U32>,fill: Element}

# The value of a target cell: what the sources give its coordinates from fill.
def placing_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,placings: Placings<Logical,Element>,index: U32) -> Element:
  Placings{+logicals,+segments,+sources,+target,+fill} = placings
  sources_value(~Element,~Logical,~original,logicals,segments,sources,Grid.coordinates(target,index),fill)

def placed_balanced(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,+count: U32,holdings: Moved.Holdings<Element>,
  +tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,placed_tree(~Element,segments,sources,grid,cells,first,count,holdings,tree)):
  match segments:
    case Nil{}: balanced
    case Con{+own,+segments}:
      match sources:
        case Nil{}: balanced
        case Con{+source,+sources}:
          match holdings:
            case Moved.Unheld{}: balanced
            case Moved.Holds{+holding,+rest}:
              +list = Tasks.placements(own,source,grid,cells,first,count)
              placed_balanced(~Element,segments,sources,grid,cells,first,count,rest,Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,tree)),depth,
                MoveParts.parts_balanced(~Element,list,holding,tree,depth,balanced))

def placed_capacity(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,+count: U32,holdings: Moved.Holdings<Element>,
  +tree: Tree.Tree<Element>) -> {Tree.capacity(Element,placed_tree(~Element,segments,sources,grid,cells,first,count,holdings,tree)) == Tree.capacity(Element,tree) : U32}:
  match segments:
    case Nil{}: {==}
    case Con{+own,+segments}:
      match sources:
        case Nil{}: {==}
        case Con{+source,+sources}:
          match holdings:
            case Moved.Unheld{}: {==}
            case Moved.Holds{+holding,+rest}:
              +list = Tasks.placements(own,source,grid,cells,first,count)
              +ran = Held.ran_tree(~Element,MoveParts.parts_ran(~Element,list,holding,tree))
              Equal.trans(U32,Tree.capacity(Element,placed_tree(~Element,segments,sources,grid,cells,first,count,rest,ran)),Tree.capacity(Element,ran),Tree.capacity(Element,tree),
                placed_capacity(~Element,segments,sources,grid,cells,first,count,rest,ran),MoveParts.parts_capacity(~Element,list,holding,tree))

# A cell of a target piece of a placing holds the value the sources give its
# coordinates from fill.
def placing_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+holdings: Moved.Holdings<Element>,+limits: Machine.Limits,+cells: List<&2,U32>,
  +piece: U32,+fill: Element,+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(cells,Grid.piece_target())))) == True{} : Bool},
  +ready: PlacedReady(~Element,~Logical,~original,logicals,segments,sources,holdings,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),Machine.half_word(limits),word,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill))),+index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece))) == True{} : Bool}) ->
  Content.At(~Element,~Placings<Logical,Element>,~placing_value(~Element,~Logical,~original),Placings{logicals,segments,sources,cells,fill},Grid.start(Grid.grid(cells,Grid.piece_target()),piece),placed_tree(~Element,segments,sources,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),holdings,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill)),U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),index):
  +grid = Grid.grid(cells,Grid.piece_target())
  +first = Grid.start(grid,piece)
  +count = Grid.span(grid,piece)
  +boxes = Grid.box_of(grid,cells,count)
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +capacity = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +elements = ShapeLaws.elements(cells)
  +single = room
  +steps = Tasks.strides_of(cells)
  +boxed = Box.grid_piece_box(limits,cells,steps,Grid.piece_target(),single,nonempty,rank_room,piece,inside,index,index_in)
  +box_cells = Box.piece_box_cells(cells,steps,boxes,U32.to_nat(first),U32.to_nat(count),index,boxed)
  +span_le = {%box_cells : {Nat.is_le(_,elements) == True{} : Bool} ViewGather.box_within_of(cells,steps,boxes,U32.to_nat(first),U32.to_nat(count),index,boxed) :
    {Nat.is_le(U32.to_nat(count),elements) == True{} : Bool}}
  +count_room = Order.transitive(U32.to_nat(count),elements,U32.to_nat(capacity),span_le,single)
  +allocatable = Order.word_le(count,capacity,1073741824,U32Division.nat_le_native(count,capacity,count_room),Limits.capacity_allocatable(limits))
  +sized = Depth.bounded(count,allocatable)
  +depth_ok = Depth.allocation_depth(count,sized)
  +capacity_exact = Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))
  +fresh_ok = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,Order.transitive(depth,30n,31n,depth_ok,{==})) : {U32.is_le(Tree.capacity(Element,fresh),2147483648) == True{} : Bool}}
  +fresh_room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  +index_room = Order.transitive(index,U32.to_nat(count),U32.to_nat(capacity),Bounds.strict_le(index,U32.to_nat(count),index_in),count_room)
  +back = Bounds.roundtrip(index,capacity,cap_ok,index_room)
  +query = U32.from_nat(index)
  +query_count = {%Equal.sym(Nat,U32.to_nat(query),index,back) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} index_in :
    {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}}
  +query_in = {%Equal.sym(Nat,ShapeLaws.elements(boxes),U32.to_nat(count),box_cells) : {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool} query_count :
    {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(boxes)) == True{} : Bool}}
  +query_inside = U32Division.nat_lt_native(query,Tree.capacity(Element,fresh),
    Bounds.nat_trans(U32.to_nat(query),U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh)),query_count,fresh_room))
  +piece_room = Limits.double_bound(limits,ShapeLaws.elements(boxes),Order.transitive(ShapeLaws.elements(boxes),elements,U32.to_nat(capacity),
    ViewGather.box_within_of(cells,steps,boxes,U32.to_nat(first),U32.to_nat(count),index,boxed),single))
  +cell = placed_cell(~Element,~Logical,~original,logicals,segments,sources,holdings,grid,cells,first,count,Machine.half_word(limits),Limits.half_word_fits(limits),word,word_ok,fresh,depth,Shape.fresh(Element,depth,fill),depth_ok,
    Tree.capacity(Element,fresh),{==},fresh_ok,piece_room,query,query_in,query_inside,fill,Readback.read_fresh(~Element,depth,fill,query),ready)
  +located = piece_positions(cells,boxes,first,U32.to_nat(count),index,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,elements,single),Digits.box_of_length(grid,cells,count),
    Box.box_end_of(cells,steps,boxes,U32.to_nat(first),U32.to_nat(count),index,boxed),
    Digits.grid_boxes(List.length(&2,U32,cells),limits,cells,Grid.piece_target(),single,nonempty,rank_room,piece,inside,index,index_in,0n))
  +corner = Grid.coordinates(cells,first)
  KernelGemm.Both{index_in,
    Equal.trans(Element,Tree.read(Element,placed_tree(~Element,segments,sources,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),holdings,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill)),query),
      sources_value(~Element,~Logical,~original,logicals,segments,sources,Placements.positions(corner,Targets.cell_coordinates(boxes,U32.to_nat(query))),fill),
      sources_value(~Element,~Logical,~original,logicals,segments,sources,Grid.coordinates(cells,U32.from_nat(Nat.add(U32.to_nat(first),index))),fill),cell,
      Equal.cong(List<&2,U32>,Element,coordinates => sources_value(~Element,~Logical,~original,logicals,segments,sources,coordinates,fill),
        Placements.positions(corner,Targets.cell_coordinates(boxes,U32.to_nat(query))),Grid.coordinates(cells,U32.from_nat(Nat.add(U32.to_nat(first),index))),
        Equal.trans(List<&2,U32>,Placements.positions(corner,Targets.cell_coordinates(boxes,U32.to_nat(query))),Placements.positions(corner,Targets.cell_coordinates(boxes,index)),
          Grid.coordinates(cells,U32.from_nat(Nat.add(U32.to_nat(first),index))),
          Equal.cong(Nat,List<&2,U32>,number => Placements.positions(corner,Targets.cell_coordinates(boxes,number)),U32.to_nat(query),index,back),located)))}

# Every cell of a target piece of a placing from offset on holds the value
# the sources give its coordinates.
def placing_cells(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+holdings: Moved.Holdings<Element>,+limits: Machine.Limits,+cells: List<&2,U32>,
  +piece: U32,+fill: Element,+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(cells,Grid.piece_target())))) == True{} : Bool},
  +ready: PlacedReady(~Element,~Logical,~original,logicals,segments,sources,holdings,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),Machine.half_word(limits),word,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill))),remaining: Nat,+offset: Nat,+left: {Nat.is_le(Nat.add(offset,remaining),U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece))) == True{} : Bool}) ->
  Members.All(Nat,{number => Content.At(~Element,~Placings<Logical,Element>,~placing_value(~Element,~Logical,~original),Placings{logicals,segments,sources,cells,fill},Grid.start(Grid.grid(cells,Grid.piece_target()),piece),placed_tree(~Element,segments,sources,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),holdings,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill)),U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),number) : Nat -> Data},Ranges.range(remaining,offset)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{placing_at(~Element,~Logical,~original,logicals,segments,sources,holdings,limits,cells,piece,fill,word,word_ok,room,nonempty,rank_room,inside,ready,offset,Order.room_index(offset,rest,U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),left)),
        placing_cells(~Element,~Logical,~original,logicals,segments,sources,holdings,limits,cells,piece,fill,word,word_ok,room,nonempty,rank_room,inside,ready,rest,1n+offset,Order.advance_room(offset,rest,U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),left))}

# A target piece fits the fresh tree it starts as.
def fresh_holds(~Element: Data,+limits: Machine.Limits,+cells: List<&2,U32>,+piece: U32,+fill: Element,
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(cells,Grid.piece_target())))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),U32.to_nat(Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill)))) == True{} : Bool}:
  +count = Grid.span(Grid.grid(cells,Grid.piece_target()),piece)
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +capacity = Machine.capacity(limits)
  +count_room = Order.transitive(U32.to_nat(count),ShapeLaws.elements(cells),U32.to_nat(capacity),Box.grid_span_room(limits,cells,Grid.piece_target(),room,nonempty,piece,inside),room)
  +allocatable = Order.word_le(count,capacity,1073741824,U32Division.nat_le_native(count,capacity,count_room),Limits.capacity_allocatable(limits))
  +sized = Depth.bounded(count,allocatable)
  {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}

# A target piece of a placing holds the value the sources give every cell,
# as many cells as the piece, and the allocator can own it.
def placing_target(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+holdings: Moved.Holdings<Element>,+limits: Machine.Limits,+cells: List<&2,U32>,
  +piece: U32,+fill: Element,+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(cells,Grid.piece_target())))) == True{} : Bool},
  +ready: PlacedReady(~Element,~Logical,~original,logicals,segments,sources,holdings,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),Machine.half_word(limits),word,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(cells,Grid.piece_target()),piece)),fill)))) ->
  Placed.PieceHolds(~Element,~Placings<Logical,Element>,~placing_value(~Element,~Logical,~original),Placings{logicals,segments,sources,cells,fill},Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),
    Buffer.content(~Element,Moved.gathered_output(~Element,placed_model(~Element,segments,sources,Grid.grid(cells,Grid.piece_target()),cells,Grid.start(Grid.grid(cells,Grid.piece_target()),piece),Grid.span(Grid.grid(cells,Grid.piece_target()),piece),holdings,ViewGather.fresh_output(~Element,fill,Grid.span(Grid.grid(cells,Grid.piece_target()),piece)))))):
  +grid = Grid.grid(cells,Grid.piece_target())
  +first = Grid.start(grid,piece)
  +count = Grid.span(grid,piece)
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +capacity = Machine.capacity(limits)
  +elements = ShapeLaws.elements(cells)
  +single = room
  +count_room = Order.transitive(U32.to_nat(count),elements,U32.to_nat(capacity),Box.grid_span_room(limits,cells,Grid.piece_target(),single,nonempty,piece,inside),single)
  +allocatable = Order.word_le(count,capacity,1073741824,U32Division.nat_le_native(count,capacity,count_room),Limits.capacity_allocatable(limits))
  +sized = Depth.bounded(count,allocatable)
  +fresh_room = fresh_holds(~Element,limits,cells,piece,fill,room,nonempty,inside)
  +tree = placed_tree(~Element,segments,sources,grid,cells,first,count,holdings,fresh)
  +tree_room = {%Equal.sym(U32,Tree.capacity(Element,tree),Tree.capacity(Element,fresh),placed_capacity(~Element,segments,sources,grid,cells,first,count,holdings,fresh)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    fresh_room : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}}
  +exact = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,ViewGather.fresh_output(~Element,fill,count)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),
    Buffer.Content{fresh,U32.to_nat(count)},Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,count)),AllocationLaws.contents(~Element,depth,fill,U32.to_nat(count)))
  +content = placed_content(~Element,segments,sources,grid,cells,first,count,holdings,ViewGather.fresh_output(~Element,fill,count),fresh,U32.to_nat(count),exact)
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Moved.gathered_output(~Element,placed_model(~Element,segments,sources,grid,cells,first,count,holdings,
      ViewGather.fresh_output(~Element,fill,count)))),Buffer.Content{tree,U32.to_nat(count)},content) :
    Placed.PieceHolds(~Element,~Placings<Logical,Element>,~placing_value(~Element,~Logical,~original),Placings{logicals,segments,sources,cells,fill},first,count,_)
  KernelGemm.Both{placing_cells(~Element,~Logical,~original,logicals,segments,sources,holdings,limits,cells,piece,fill,word,word_ok,room,nonempty,rank_room,inside,ready,U32.to_nat(count),0n,Order.reflexive(U32.to_nat(count))),
    KernelGemm.Both{{==},Buffer.ContentReady{depth,placed_balanced(~Element,segments,sources,grid,cells,first,count,holdings,fresh,depth,Shape.fresh(Element,depth,fill)),
      Depth.allocation_depth(count,sized),tree_room}}}
