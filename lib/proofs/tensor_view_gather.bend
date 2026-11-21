# Tensors.view_gather as a gather of Tensors.moved_with: a target piece of a
# view reads its cells through one part of the held source pieces into a
# fresh buffer, and the holding keeps the order of its pieces.
import Base
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../tensor_move.bend as Move
import ../tensor_tasks.bend as Tasks
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./tensor_move_part.bend as Part
import ./tensor_move_parts.bend as Parts
import ./tensor_moved.bend as Moved
import ./tensor_placed.bend as Placed
import ./tensor_view_part.bend as ViewPart
import ./u32_division.bend as U32Division
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_pieces.bend as Laws
import ./tensor_taken_stack.bend as Taken
import ./tensor_task_runs.bend as Runs
import ./tensor_pieces_model.bend as PiecesModel
import ./boolean_logic.bend as Logic
import ./tensor_components.bend as Components
import ./tensor_box_cells.bend as BoxCells
import ./tensor_target_cells.bend as Targets
import ./tensor_view_reads.bend as ViewReads
import ../storage_allocation.bend as Allocation
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./tensor_grid_box.bend as Box
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_pieces_content.bend as Content
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ../kernel_gemm.bend as KernelGemm
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds

# ---- The part of a target piece ----

# A view reads a target piece through the part of its box: the mirrored
# axes restricted to the box, from the offset of the piece view.
def piece_part(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+grid: Grid.Grid,+first: U32,+count: U32) ->
  {Tensors.view_part(flags,Grid.box_of(grid,Views.axis_extents(axes),count),
      Grid.piece_view(grid,Views.axis_extents(axes),first,count,Views.mirrored(Views.View{axes,offset},flags))) ==
    Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(grid,Views.axis_extents(axes),count),flags),
      ViewPart.view_offset(Grid.piece_view(grid,Views.axis_extents(axes),first,count,Views.mirrored(Views.View{axes,offset},flags))),0} : Move.Part}:
  +view = Grid.piece_view(grid,Views.axis_extents(axes),first,count,Views.mirrored(Views.View{axes,offset},flags))
  +boxes = Grid.box_of(grid,Views.axis_extents(axes),count)
  +mirrored = Views.mirrored_axes(axes,flags)
  +axes_exact = {%ViewPart.mirrored_extents(axes,flags) :
      {ViewPart.view_axes_of(Grid.restricted(mirrored,Grid.coordinates(_,first),boxes,Views.mirrored_offset(axes,flags,offset))) == ViewPart.boxed_axes(mirrored,boxes) :
        List<&2,Views.Axis>}
    ViewPart.restricted_axes(mirrored,boxes,first,Views.mirrored_offset(axes,flags,offset)) :
    {ViewPart.view_axes_of(view) == ViewPart.boxed_axes(mirrored,boxes) : List<&2,Views.Axis>}}
  %ViewPart.view_eta(view) : {Tensors.view_part(flags,boxes,_) == Move.Part{ViewPart.box_axes(mirrored,boxes,flags),ViewPart.view_offset(view),0} : Move.Part}
  %Equal.sym(List<&2,Views.Axis>,ViewPart.view_axes_of(view),ViewPart.boxed_axes(mirrored,boxes),axes_exact) :
    {Move.Part{Tensors.view_axes(_,flags,Tasks.strides_of(boxes)),ViewPart.view_offset(view),0} ==
      Move.Part{ViewPart.box_axes(mirrored,boxes,flags),ViewPart.view_offset(view),0} : Move.Part}
  {==}

# ---- The model of a gather ----

def part_axes(+part: Move.Part) -> List<&2,Move.Axis>:
  Move.Part{+axes,+source,+target} = part
  axes

def part_source(+part: Move.Part) -> U32:
  Move.Part{+axes,+source,+target} = part
  source

def part_target(+part: Move.Part) -> U32:
  Move.Part{+axes,+source,+target} = part
  target

def part_eta(+part: Move.Part) -> {Move.Part{part_axes(part),part_source(part),part_target(part)} == part : Move.Part}:
  match part:
    case Move.Part{+axes,+source,+target}: {==}

def pair_eta(~Element: Data,pair: Move.Holding<Element> & Storage.Buffer<Element>) ->
  {(Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,pair),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,pair)) == pair :
    Move.Holding<Element> & Storage.Buffer<Element>}:
  (first,second) = pair
  {==}

# The holding a part leaves of a held holding, writing into a buffer tree.
def held_after(~Element: Data,+part: Move.Part,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Holding<Element>:
  Held.ran_holding(~Element,Part.part_ran(~Element,part_axes(part),part_source(part),part_target(part),holding,tree))

def view_piece(+context: Tensors.Viewed,+first: U32,+count: U32) -> Move.Part:
  Tensors.Viewed{source,+target_grid,+target,+axes,+offset,+reversed} = context
  Tensors.view_part(reversed,Grid.box_of(target_grid,target,count),Grid.piece_view(target_grid,target,first,count,Views.mirrored(Views.View{axes,offset},reversed)))

def fresh_output(~Element: Data,+fill: Element,+count: U32) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))

# The holdings and the target piece a view gathers, as a model.
def view_model(~Element: Data,+context: Tensors.Viewed,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holdings: Moved.Holdings<Element>) ->
  Moved.Gathered<Element>:
  match holdings:
    case Moved.Holds{+holding,+rest}:
      +part = view_piece(context,first,count)
      Moved.Gathered{Moved.Holds{held_after(~Element,part,holding,Buffer.tree(~Element,fresh_output(~Element,fill,count))),rest},
        Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))))}
    case Moved.Unheld{}:
      Moved.Gathered{Moved.Unheld{},fresh_output(~Element,fill,count)}

# A part of a held holding into a buffer holds what its rows leave.
def part_held(~Element: Data,+part: Move.Part,+holding: Held.Holding<Element>,+fill: Element,+count: U32) ->
  {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))) ==
    Held.pack(~Element,held_after(~Element,part,holding,Buffer.tree(~Element,fresh_output(~Element,fill,count)))) : Move.Holding<Element>}:
  +model = fresh_output(~Element,fill,count)
  +axes = part_axes(part)
  +source = part_source(part)
  +target = part_target(part)
  %part_eta(part) :
    {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,_,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))) ==
      Held.pack(~Element,held_after(~Element,part,holding,Buffer.tree(~Element,model))) : Move.Holding<Element>}
  %Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count)) :
    {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),_)) ==
      Held.pack(~Element,held_after(~Element,part,holding,Buffer.tree(~Element,model))) : Move.Holding<Element>}
  Held.run_held(~Element,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)),
    Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model),
    Part.part_refines(~Element,axes,source,target,holding,model))

# A part writes its target buffer and leaves its holding as the model says.
def written_exact(~Element: Data,+part: Move.Part,+holding: Held.Holding<Element>,+rest: Moved.Holdings<Element>,+fill: Element,+count: U32) ->
  {Tensors.view_written(~Element,Moved.pack_holdings(~Element,rest),Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))) ==
    (Moved.pack_holdings(~Element,Moved.Holds{held_after(~Element,part,holding,Buffer.tree(~Element,fresh_output(~Element,fill,count))),rest}),
      Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
        Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))))) : Tensors.Holdings<Element> & Storage.Buffer<Element>}:
  +after = held_after(~Element,part,holding,Buffer.tree(~Element,fresh_output(~Element,fill,count)))
  %pair_eta(~Element,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))) :
    {Tensors.view_written(~Element,Moved.pack_holdings(~Element,rest),_) ==
      (Tensors.Holds{Held.pack(~Element,after),Moved.pack_holdings(~Element,rest)},Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))))) :
      Tensors.Holdings<Element> & Storage.Buffer<Element>}
  %Equal.sym(Move.Holding<Element>,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))),Held.pack(~Element,after),
      part_held(~Element,part,holding,fill,count)) :
    {(Tensors.Holds{_,Moved.pack_holdings(~Element,rest)},Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))) ==
      (Tensors.Holds{Held.pack(~Element,after),Moved.pack_holdings(~Element,rest)},Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))))) :
      Tensors.Holdings<Element> & Storage.Buffer<Element>}
  %Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))) :
    {(Tensors.Holds{Held.pack(~Element,after),Moved.pack_holdings(~Element,rest)},_) ==
      (Tensors.Holds{Held.pack(~Element,after),Moved.pack_holdings(~Element,rest)},Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,part,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))))) :
      Tensors.Holdings<Element> & Storage.Buffer<Element>}
  {==}

# ---- A part into a fresh buffer ----

def fresh_length(~Element: Data,+fill: Element,+count: U32) -> {Buffer.length(~Element,fresh_output(~Element,fill,count)) == U32.to_nat(count) : Nat}:
  +depth = Tensors.depth_for(count)
  Equal.trans(Nat,Buffer.length(~Element,fresh_output(~Element,fill,count)),
    Buffer.content_length(~Element,Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))),U32.to_nat(count),
    Held.allocated_length(~Element,depth,fill,U32.to_nat(count)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count))),
      Buffer.Content{Tree.fresh(Element,depth,fill),U32.to_nat(count)},AllocationLaws.contents(~Element,depth,fill,U32.to_nat(count))))

# What a part writes into a fresh buffer of count cells.
def part_contents(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+fill: Element,+count: U32) ->
  {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))) ==
    Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Tree.fresh(Element,Tensors.depth_for(count),fill))),U32.to_nat(count)} :
    Buffer.Content<Element>}:
  +model = fresh_output(~Element,fill,count)
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  %Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count)) :
    {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),_))) ==
      Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,fresh)),U32.to_nat(count)} : Buffer.Content<Element>}
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
    Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)},
    Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,fresh)),U32.to_nat(count)},
    Held.run_target(~Element,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)),
      Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model),Part.part_refines(~Element,axes,source,target,holding,model)),
    {%Held.fresh_tree(~Element,depth,fill,U32.to_nat(count)) :
        {Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)} ==
          Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,_)),U32.to_nat(count)} : Buffer.Content<Element>}
      %fresh_length(~Element,fill,count) :
        {Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)} ==
          Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),_} : Buffer.Content<Element>}
      {==} :
      {Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model))),Buffer.length(~Element,model)} ==
        Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,fresh)),U32.to_nat(count)} : Buffer.Content<Element>}})

# A part writes a fresh buffer the allocator can own.
def part_ready(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+fill: Element,+count: U32,
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.ContentReady<Element,Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Tree.fresh(Element,Tensors.depth_for(count),fill))),
    U32.to_nat(count)}>:
  +depth = Tensors.depth_for(count)
  +sized = Depth.bounded(count,allocatable)
  +fresh = Tree.fresh(Element,depth,fill)
  +ran = Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,fresh))
  +capacity_exact = Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))
  +fresh_room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  +room = {%Equal.sym(U32,Tree.capacity(Element,ran),Tree.capacity(Element,fresh),Part.part_capacity(~Element,axes,source,target,holding,fresh)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    fresh_room : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,ran))) == True{} : Bool}}
  Buffer.ContentReady{depth,Part.part_balanced(~Element,axes,source,target,holding,fresh,depth,Shape.fresh(Element,depth,fill)),Depth.allocation_depth(count,sized),room}

def part_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+fill: Element,+count: U32,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: Part.DenseSteps(axes),
  +target_room: {Nat.is_le(Nat.add(U32.to_nat(target),Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},
  +target_fits: {Nat.is_le(Nat.add(U32.to_nat(target),Part.total(axes)),U32.to_nat(count)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: Part.StridesRoom(axes,word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +cells: Part.CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,Part.total(axes)),
  +index: Nat,+inside: {Nat.is_lt(index,Part.total(axes)) == True{} : Bool}) ->
  {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,target,holding,Tree.fresh(Element,Tensors.depth_for(count),fill))),
      (target + U32.from_nat(index) : U32)) == value(context,Part.cell_source(axes,index,source,target)) : Element}:
  +depth = Tensors.depth_for(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +sized = Depth.bounded(count,allocatable)
  +capacity_exact = Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))
  +fresh_ok = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,Order.transitive(depth,30n,31n,Depth.allocation_depth(count,sized),{==})) : {U32.is_le(Tree.capacity(Element,fresh),2147483648) == True{} : Bool}}
  +fresh_room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  Part.dense_written(~Element,~Context,~value,context,axes,source,target,holding,fresh,depth,Shape.fresh(Element,depth,fill),Depth.allocation_depth(count,sized),fresh_ok,
    cap,cap_ok,room,dense,target_room,Order.transitive(Nat.add(U32.to_nat(target),Part.total(axes)),U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh)),target_fits,fresh_room),
    word,word_ok,strides,stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,values,cells,index,inside)

# ---- A part of a covering holding ----

# A part leaves a covering holding covering its cells with their values.
def parted_covered(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,
  +fill: Element,+count: U32,-holding: Move.Holding<Element>,+low: Nat,+high: Nat,+layout: Held.Spans,
  +covered: Held.Covered<Element,Context,value,context,holding,low,high,layout>) ->
  Held.Covered<Element,Context,value,context,
    Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},holding,Tensors.buffer(~Element,fill,count))),low,high,layout>:
  Held.Covered{+model,+packed,+tiled,+values,+ordered} = covered
  +tree = Buffer.tree(~Element,fresh_output(~Element,fill,count))
  +ran = Held.ran_holding(~Element,Part.part_ran(~Element,axes,source,target,model,tree))
  +order_exact = Part.part_order(~Element,axes,source,target,model,tree)
  %Equal.sym(Move.Holding<Element>,holding,Held.pack(~Element,model),packed) :
    Held.Covered<Element,Context,value,context,
      Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,target},_,Tensors.buffer(~Element,fill,count))),low,high,layout>
  Held.Covered{ran,part_held(~Element,Move.Part{axes,source,target},model,fill,count),
    Held.order_tiled(~Element,ran,low,high,
      %Equal.sym(Held.Stack<Element>,Held.order(~Element,ran),Held.order(~Element,model),order_exact) : Held.Tiles(~Element,_,low,high)
      Held.order_tiles(~Element,model,low,high,tiled)),
    Part.part_values(~Element,~Context,~value,context,axes,source,target,model,tree,values),
    Equal.trans(Held.Spans,Held.spans(~Element,Held.order(~Element,ran)),Held.spans(~Element,Held.order(~Element,model)),layout,
      Equal.cong(Held.Stack<Element>,Held.Spans,stack => Held.spans(~Element,stack),Held.order(~Element,ran),Held.order(~Element,model),order_exact),ordered)}

# A cell inside [low, high).
type Between<-low: Nat,-high: Nat,-cell: Nat> is Data:
  Between{after: {Nat.is_le(low,cell) == True{} : Bool},before: {Nat.is_lt(cell,high) == True{} : Bool}}

def between_held(~Element: Data,+stack: Held.Stack<Element>,+low: Nat,+high: Nat,+cell: Nat,+tiles: Held.Tiles(~Element,stack,low,high),+between: Between<low,high,cell>) ->
  Held.SpanHeld(Held.spans(~Element,stack),cell):
  Between{+after,+before} = between
  Held.tiles_member(~Element,stack,low,high,cell,tiles,after,before)

# A context whose cell index is the cell its part reads into target cell
# index.
type Parted<-Context: Data> is Data:
  Parted{context: Context,axes: List<&2,Move.Axis>,source: U32}

def parted_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,parted: Parted<Context>,index: U32) -> Element:
  Parted{context,+axes,+source} = parted
  value(context,Part.offsets_source(Move.row_offsets(axes,index,source,0)))

# What a dense part of a covering holding into a buffer of its own needs, for
# every cell of it.
type PartCells<-Element: Data,-Context: Data,-value: Context -> U32 -> Element,-context: Context> is Data:
  PartCells{holding: Held.Holding<Element>,axes: List<&2,Move.Axis>,source: U32,fill: Element,count: U32,
    allocatable: {U32.is_le(count,1073741824) == True{} : Bool},total_exact: {Part.total(axes) == U32.to_nat(count) : Nat},
    cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},
    dense: Part.DenseSteps(axes),word: U32,word_ok: {U32.is_le(word,2147483648) == True{} : Bool},strides: Part.StridesRoom(axes,word),
    low: Nat,high: Nat,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    tiled: Held.Tiled(~Element,holding,low,high),values: Held.HoldingValues(~Element,~Context,~value,context,holding),
    held: Part.CellsHeld(axes,source,0,Held.spans(~Element,Held.order(~Element,holding)),0n,Part.total(axes))}

def part_size(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: PartCells<Element,Context,value,context>) -> Nat:
  PartCells{holding,axes,source,fill,+count,allocatable,total_exact,cap,cap_ok,room,dense,word,word_ok,strides,low,high,capacity,capacity_ok,bounded,tiled,values,held} = cells
  U32.to_nat(count)

def PartAt(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+cells: PartCells<Element,Context,value,context>,+index: Nat) -> Data:
  PartCells{+holding,+axes,+source,+fill,+count,allocatable,total_exact,cap,cap_ok,room,dense,word,word_ok,strides,low,high,capacity,capacity_ok,bounded,tiled,values,held} = cells
  Content.At(~Element,~Parted<Context>,~parted_value(~Element,~Context,~value),Parted{context,axes,source},0,
    Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,0,holding,Tree.fresh(Element,Tensors.depth_for(count),fill))),U32.to_nat(count),index)

def part_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+cells: PartCells<Element,Context,value,context>,+index: Nat,
  +inside: {Nat.is_lt(index,part_size(~Element,~Context,~value,~context,cells)) == True{} : Bool}) -> PartAt(~Element,~Context,~value,context,cells,index):
  PartCells{+holding,+axes,+source,+fill,+count,+allocatable,+total_exact,+cap,+cap_ok,+room,+dense,+word,+word_ok,+strides,+low,+high,+capacity,+capacity_ok,+bounded,
    +tiled,+values,+held} = cells
  +stack = Held.order(~Element,holding)
  KernelGemm.Both{inside,
    %Arithmetic.add_zero_left(U32.from_nat(index)) :
      {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,0,holding,Tree.fresh(Element,Tensors.depth_for(count),fill))),_) ==
        value(context,Part.cell_source(axes,index,source,0)) : Element}
    part_read(~Element,~Context,~value,context,axes,source,0,holding,fill,count,allocatable,cap,cap_ok,room,dense,
      Order.transitive(Part.total(axes),Nat.double(Part.total(axes)),U32.to_nat(cap),Order.le_double(Part.total(axes)),room),
      %total_exact : {Nat.is_le(Part.total(axes),_) == True{} : Bool} Order.reflexive(Part.total(axes)),
      word,word_ok,strides,stack,{==},low,high,capacity,capacity_ok,bounded,
      Held.tiles_sorted(~Element,stack,low,high,Held.order_tiles(~Element,holding,low,high,tiled)),values,held,index,
      %Equal.sym(Nat,Part.total(axes),U32.to_nat(count),total_exact) : {Nat.is_lt(index,_) == True{} : Bool} inside)}

# A dense part of a covering holding into a buffer of its own, held as the only
# piece, covers its cells with the cells the part reads.
def parted_alone(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Held.Holding<Element>,+axes: List<&2,Move.Axis>,
  +source: U32,+fill: Element,+count: U32,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+total_exact: {Part.total(axes) == U32.to_nat(count) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},
  +dense: Part.DenseSteps(axes),+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: Part.StridesRoom(axes,word),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Held.Tiled(~Element,holding,low,high),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +held: Part.CellsHeld(axes,source,0,Held.spans(~Element,Held.order(~Element,holding)),0n,Part.total(axes))) ->
  Held.Covered<Element,Parted<Context>,parted_value(~Element,~Context,~value),Parted{context,axes,source},
    Move.held(~Element,Partition.Leaf{Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.part(~Element,Move.Part{axes,source,0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))))),0,count,Unit{}}),
    0n,U32.to_nat(count),Held.layout_spans(Pieces.Range{0,count},Held.Ended{})>:
  +ran = Held.ran_tree(~Element,Part.part_ran(~Element,axes,source,0,holding,Tree.fresh(Element,Tensors.depth_for(count),fill)))
  Held.alone_content(~Element,~Parted<Context>,~parted_value(~Element,~Context,~value),Parted{context,axes,source},
    Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))),
    count,ran,
    Equal.trans(Buffer.Content<Element>,
      Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))))),
      Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))),
      Buffer.Content{ran,U32.to_nat(count)},
      Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{axes,source,0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count)))),
      part_contents(~Element,axes,source,0,holding,fill,count)),
    part_ready(~Element,axes,source,0,holding,fill,count,allocatable),
    Members.range_all(~PartCells<Element,Context,value,context>,~PartAt(~Element,~Context,~value,context),~part_size(~Element,~Context,~value,~context),
      ~part_at(~Element,~Context,~value,context),U32.to_nat(count),0n,
      PartCells{holding,axes,source,fill,count,allocatable,total_exact,cap,cap_ok,room,dense,word,word_ok,strides,low,high,capacity,capacity_ok,bounded,tiled,values,held},
      Order.reflexive(U32.to_nat(count))))

# ---- The values of a target piece ----

# The values a view reads: the source values at the view address of every
# target cell.
type Viewing<-Logical: Data> is Data:
  Viewing{logical: Logical,axes: List<&2,Views.Axis>,offset: U32,flags: List<&2,Bool>}

def viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,viewing: Viewing<Logical>,index: U32) -> Element:
  Viewing{+logical,+axes,+offset,+flags} = viewing
  original(logical,(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),
    U32.to_nat(index)) : U32))

# ---- Views that reverse no axis ----

# Flags that reverse no axis.
def forward(flags: List<&2,Bool>) -> Bool:
  match flags:
    case Nil{}: True{}
    case Con{+flag,+rest}: Bool.not(flag) && forward(rest)

def first_forward(flags: List<&2,Bool>,+none: {forward(flags) == True{} : Bool}) -> {Views.flag(flags) == False{} : Bool}:
  match flags:
    case Nil{}: {==}
    case Con{+flag,+rest}: Logic.not_true(flag,Logic.and_true_left(Bool.not(flag),forward(rest),none))

def later_forward(flags: List<&2,Bool>,+none: {forward(flags) == True{} : Bool}) -> {forward(Views.later_flags(flags)) == True{} : Bool}:
  match flags:
    case Nil{}: {==}
    case Con{+flag,+rest}: Logic.and_true_right(Bool.not(flag),forward(rest),none)

def forward_append(flags: List<&2,Bool>,+none: {forward(flags) == True{} : Bool}) -> {forward(List.append(&2,Bool,flags,[False{}])) == True{} : Bool}:
  match flags:
    case Nil{}: {==}
    case Con{+flag,+rest}:
      Logic.and_true(Bool.not(flag),forward(List.append(&2,Bool,rest,[False{}])),Logic.and_true_left(Bool.not(flag),forward(rest),none),
        forward_append(rest,Logic.and_true_right(Bool.not(flag),forward(rest),none)))

def forward_axes(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+none: {forward(flags) == True{} : Bool}) -> {Views.mirrored_axes(axes,flags) == axes : List<&2,Views.Axis>}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+tail}:
      %Equal.sym(Bool,Views.flag(flags),False{},first_forward(flags,none)) :
        {Views.Axis{extent,Views.directed(_,stride)} <> Views.mirrored_axes(tail,Views.later_flags(flags)) == Views.Axis{extent,stride} <> tail : List<&2,Views.Axis>}
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,rest => Views.Axis{extent,stride} <> rest,Views.mirrored_axes(tail,Views.later_flags(flags)),tail,
        forward_axes(tail,Views.later_flags(flags),later_forward(flags,none)))

def forward_offset(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+none: {forward(flags) == True{} : Bool}) ->
  {Views.mirrored_offset(axes,flags,offset) == offset : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+tail}:
      %Equal.sym(Bool,Views.flag(flags),False{},first_forward(flags,none)) :
        {Views.mirrored_offset(tail,Views.later_flags(flags),Views.mirrored_start(_,extent,stride,offset)) == offset : U32}
      forward_offset(tail,Views.later_flags(flags),offset,later_forward(flags,none))

# A view whose flags reverse no axis reads as the view without flags.
def forward_viewed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axes: List<&2,Views.Axis>,+offset: U32,
  +flags: List<&2,Bool>,+none: {forward(flags) == True{} : Bool},+index: U32) ->
  {viewed(~Element,~Logical,~original,Viewing{logical,axes,offset,flags},index) == viewed(~Element,~Logical,~original,Viewing{logical,axes,offset,[]},index) : Element}:
  %Equal.sym(U32,Views.mirrored_offset(axes,flags,offset),offset,forward_offset(axes,flags,offset,none)) :
    {original(logical,(_ + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),U32.to_nat(index)) : U32)) ==
      viewed(~Element,~Logical,~original,Viewing{logical,axes,offset,[]},index) : Element}
  %Equal.sym(List<&2,Views.Axis>,Views.mirrored_axes(axes,flags),axes,forward_axes(axes,flags,none)) :
    {original(logical,(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(_),U32.to_nat(index)) : U32)) ==
      viewed(~Element,~Logical,~original,Viewing{logical,axes,offset,[]},index) : Element}
  %Equal.sym(U32,Views.mirrored_offset(axes,[],offset),offset,forward_offset(axes,[],offset,{==})) :
    {original(logical,(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(index)) : U32)) ==
      original(logical,(_ + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),U32.to_nat(index)) : U32)) : Element}
  %Equal.sym(List<&2,Views.Axis>,Views.mirrored_axes(axes,[]),axes,forward_axes(axes,[],{==})) :
    {original(logical,(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(index)) : U32)) ==
      original(logical,(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(_),U32.to_nat(index)) : U32)) : Element}
  {==}

def box_within_of(+extents: List<&2,U32>,+steps: List<&2,U32>,+boxes: List<&2,U32>,+start: Nat,+cells: Nat,+index: Nat,
  +boxed: Box.PieceBox(extents,steps,boxes,start,cells,index)) -> {Nat.is_le(ShapeLaws.elements(boxes),ShapeLaws.elements(extents)) == True{} : Bool}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  box_le

# The part of a target piece runs over as many box indices as the piece has
# cells.
def piece_total(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+grid: Grid.Grid,+count: U32,+cells: Nat,
  +exact: {ShapeLaws.elements(Grid.box_of(grid,Views.axis_extents(axes),count)) == cells : Nat}) ->
  {Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(grid,Views.axis_extents(axes),count),flags)) == cells : Nat}:
  +mirrored = Views.mirrored_axes(axes,flags)
  +boxes = Grid.box_of(grid,Views.axis_extents(axes),count)
  +moves = ViewPart.box_axes(mirrored,boxes,flags)
  +same = {%ViewPart.mirrored_extents(axes,flags) : ViewPart.SameLength(mirrored,Grid.box_of(grid,_,count)) ViewPart.same_box_of(mirrored,grid,count) :
    ViewPart.SameLength(mirrored,boxes)}
  Equal.trans(Nat,Part.total(moves),ShapeLaws.elements(ViewPart.move_extents(moves)),cells,ViewPart.total_elements(moves),
    Equal.trans(Nat,ShapeLaws.elements(ViewPart.move_extents(moves)),ShapeLaws.elements(boxes),cells,
      Equal.cong(List<&2,U32>,Nat,value => ShapeLaws.elements(value),ViewPart.move_extents(moves),boxes,ViewPart.box_extents(mirrored,boxes,flags,same)),exact))

# Cell index of a target piece of a view holds the source value at the view
# address of the cell.
def view_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,
  +offset: U32,+target: U32,+piece: U32,+holding: Held.Holding<Element>,+fill: Element,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+allocatable: {U32.is_le(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),1073741824) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: Part.StridesRoom(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,
  +bounded: {Nat.is_le(end,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+sorted: Held.Sorted(~Element,stack,start,end),
  +values: Held.HoldingValues(~Element,~Logical,~original,logical,holding),+cells: Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,Held.spans(~Element,stack),0n,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  Content.At(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),index):
  +mirrored = Views.mirrored_axes(axes,flags)
  +steps = ViewPart.view_steps(mirrored)
  +span = U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))
  +half = Machine.half_word(limits)
  +boxed = Box.grid_piece_box(limits,Views.axis_extents(axes),steps,target,room,nonempty,rank_room,piece,inside,index,index_in)
  +box_cells = Box.piece_box_cells(Views.axis_extents(axes),steps,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),span,index,boxed)
  +total_exact = piece_total(axes,flags,Grid.grid(Views.axis_extents(axes),target),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),span,box_cells)
  +span_room = {%box_cells : {Nat.is_le(_,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool} box_within_of(Views.axis_extents(axes),steps,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),span,index,boxed) :
    {Nat.is_le(span,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +whole_room = LimitBounds.double_bound(limits,span,Order.transitive(span,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),span_room,room))
  +part_room = {%Equal.sym(Nat,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),span,total_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(half)) == True{} : Bool} whole_room :
    {Nat.is_le(Nat.double(Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags))),U32.to_nat(half)) == True{} : Bool}}
  +same = {%ViewPart.mirrored_extents(axes,flags) : ViewPart.SameLength(mirrored,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),_,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) ViewPart.same_box_of(mirrored,Grid.grid(Views.axis_extents(axes),target),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)) :
    ViewPart.SameLength(mirrored,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))}
  +address = Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)
  +address_room = Order.transitive(address,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),
    Bounds.strict_le(address,ShapeLaws.elements(Views.axis_extents(axes)),Box.box_end_of(Views.axis_extents(axes),steps,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),span,index,boxed)),room)
  +back = Bounds.roundtrip(address,Machine.capacity(limits),LimitBounds.capacity_fits(limits),address_room)
  KernelGemm.Both{index_in,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(address)),address,back) :
        {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.from_nat(index)) == original(logical,(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),steps,_) : U32)) : Element}
      %ViewPart.view_piece_source(limits,axes,flags,offset,target,room,nonempty,rank_room,piece,inside,index,index_in) :
        {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.from_nat(index)) == original(logical,_) : Element}
      %Arithmetic.add_zero_left(U32.from_nat(index)) : {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),_) == original(logical,Part.cell_source(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),index,ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0)) : Element}
      part_read(~Element,~Logical,~original,logical,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),allocatable,half,LimitBounds.half_word_fits(limits),part_room,
        ViewPart.box_dense(mirrored,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags,same),Part.single_room(Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),half,part_room),
        {%Equal.sym(Nat,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),span,total_exact) : {Nat.is_le(_,span) == True{} : Bool} Order.reflexive(span) : {Nat.is_le(Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),span) == True{} : Bool}},
        word,word_ok,strides,stack,ordered,start,end,Machine.capacity(limits),LimitBounds.capacity_fits(limits),bounded,sorted,values,{%Equal.sym(Nat,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),span,total_exact) : Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,Held.spans(~Element,stack),0n,_) cells : Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,Held.spans(~Element,stack),0n,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)))},index,
        {%Equal.sym(Nat,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags)),span,total_exact) : {Nat.is_lt(index,_) == True{} : Bool} index_in : {Nat.is_lt(index,Part.total(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags))) == True{} : Bool}}) :
      {Tree.read(Element,Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.from_nat(index)) == viewed(~Element,~Logical,~original,Viewing{logical,axes,offset,flags},U32.from_nat(address)) : Element}}}

# Every cell of a target piece of a view holds the source value at its view
# address.
def view_cells(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+position: Nat,+logical: Logical,+limits: Machine.Limits,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+piece: U32,+holding: Held.Holding<Element>,+fill: Element,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+allocatable: {U32.is_le(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),1073741824) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: Part.StridesRoom(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,
  +bounded: {Nat.is_le(end,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+sorted: Held.Sorted(~Element,stack,start,end),
  +values: Held.HoldingValues(~Element,~Logical,~original,logical,holding),+cells: Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,Held.spans(~Element,stack),0n,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),
  +left: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  Members.All(Nat,{index => Content.At(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),index) : Nat -> Data},
    Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{view_at(~Element,~Logical,~original,logical,limits,axes,flags,offset,target,piece,holding,fill,room,nonempty,rank_room,inside,allocatable,
          word,word_ok,strides,stack,ordered,start,end,bounded,sorted,values,cells,position,Order.room_index(position,rest,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),left)),
        view_cells(~Element,~Logical,~original,rest,1n+position,logical,limits,axes,flags,offset,target,piece,holding,fill,room,nonempty,rank_room,inside,allocatable,
          word,word_ok,strides,stack,ordered,start,end,bounded,sorted,values,cells,Order.advance_room(position,rest,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),left))}

# The buffer a view gathers for a target piece holds the source value at
# the view address of every cell, as many cells as the piece, and the
# allocator can own it.
def view_target(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+piece: U32,+holding: Held.Holding<Element>,+fill: Element,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+allocatable: {U32.is_le(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),1073741824) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: Part.StridesRoom(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,
  +bounded: {Nat.is_le(end,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+sorted: Held.Sorted(~Element,stack,start,end),
  +values: Held.HoldingValues(~Element,~Logical,~original,logical,holding),+cells: Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,Held.spans(~Element,stack),0n,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))) ->
  Placed.PieceHolds(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),
    Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,view_piece(Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))))):
  %Equal.sym(Move.Part,view_piece(Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0},piece_part(axes,flags,offset,Grid.grid(Views.axis_extents(axes),target),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) :
    Placed.PieceHolds(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,_,Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))))))
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))))),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))),
      Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))))) :
    Placed.PieceHolds(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),_)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.part(~Element,Move.Part{ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0},Held.pack(~Element,holding),Tensors.buffer(~Element,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))))),Buffer.Content{Held.ran_tree(~Element,Part.part_ran(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,Tree.fresh(Element,Tensors.depth_for(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),fill))),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))},
      part_contents(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) :
    Placed.PieceHolds(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),_)
  KernelGemm.Both{view_cells(~Element,~Logical,~original,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),0n,logical,limits,axes,flags,offset,target,piece,holding,fill,room,nonempty,rank_room,inside,
      allocatable,word,word_ok,strides,stack,ordered,start,end,bounded,sorted,values,cells,Order.reflexive(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)))),
    KernelGemm.Both{{==},part_ready(~Element,ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,holding,fill,Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),allocatable)}}

# ---- The source cells a target piece of a view reads are held ----

def read_exact_of(+offset: U32,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+address: U32,+length: U32,
  +read: ViewReads.ReadAt(offset,axes,flags,index,address,length)) -> {U32.to_nat(address) == Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,index)) : Nat}:
  Held.Both{+exact,+below} = read
  exact

def read_below_of(+offset: U32,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+address: U32,+length: U32,
  +read: ViewReads.ReadAt(offset,axes,flags,index,address,length)) ->
  {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,index)),U32.to_nat(length)) == True{} : Bool}:
  Held.Both{+exact,+below} = read
  below

# What a separable view needs of a component: its source row reads inside
# the words, its classes end inside their axes and cover the target pieces.
def ViewSeparate(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32) -> Data:
  Logic.When(Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>)

# A source cell inside a box of the source lies in a piece the box lists.
def source_cell_held(+limits: Machine.Limits,+source: List<&2,U32>,
  +room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+box: List<&2,Tasks.Span>,+value: Nat,
  +within: Components.SpansIn(box,source),+found: ViewReads.InSource(box,source,value)) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,box),Grid.grid(source,Grid.piece_target())),value):
  Held.Both{+inside,+below} = found
  BoxCells.cell_held(limits,source,Grid.piece_target(),room,nonempty,rank_room,box,value,within,inside,below)

def cell_separate(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,+piece: U32,
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(source)) == True{} : Bool},+fit: ViewReads.AxesFit(source,Grid.coordinates(source,offset),axes,Machine.half_word(limits)),
  +blocks: ViewReads.BlocksIn(Grid.grid(source,Grid.piece_target()),source,0,Machine.half_word(limits)),+separate: {Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]) == True{} : Bool},+facts: Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>,
  +member: Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),component),piece),+index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool},
  +address_below: {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),ShapeLaws.elements(source)) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +at_in: {Nat.is_lt(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)))):
  Held.Both{Held.Both{+rooms,Held.Both{+axes_ok,+axes_below}},Held.Both{+ends,+classes_ok}} = facts
  +cap_half = Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),LimitBounds.capacity_within_half(limits))
  +classes_in = Targets.target_classes(limits,Views.axis_extents(axes),target,room,nonempty,rank_room,[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],component,separate,classes_ok,piece,member,index,index_in)
  +held = {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),True{},separate) : ViewReads.ClassesWhen(_,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),axes,Targets.cell_coordinates(Views.axis_extents(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))) classes_in :
    ViewReads.ClassesWhen(Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),axes,Targets.cell_coordinates(Views.axis_extents(axes),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)))}
  +found = ViewReads.view_source_box(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),offset,component,Machine.half_word(limits),LimitBounds.half_word_fits(limits),held,
    Order.transitive(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),source_rank,cap_half),
    LimitBounds.double_bound(limits,ShapeLaws.elements(source),source_room),source_positive,offset_in,target_room,at_in,fit,axes_ok,blocks,address_below)
  +within = {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),True{},separate) : Components.SpansIn(Tasks.box_when(_,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(source,Grid.piece_target()),source,offset),source)
    BoxCells.boxed_coordinates_in(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(source,Grid.piece_target()),source,offset,Machine.half_word(limits),LimitBounds.half_word_fits(limits),rooms,ends,axes_ok) : Components.SpansIn(Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component),source)}
  source_cell_held(limits,source,source_room,source_positive,source_rank,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),within,found)

def cell_whole(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,+piece: U32,
  +index: Nat,+source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +whole: {Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]) == False{} : Bool},+address_below: {Nat.is_lt(Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),ShapeLaws.elements(source)) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)))):
  +within = {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),False{},whole) : Components.SpansIn(Tasks.box_when(_,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(source,Grid.piece_target()),source,offset),source)
    Components.whole_in(source) : Components.SpansIn(Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component),source)}
  +found = {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),False{},whole) : ViewReads.InSource(Tasks.box_when(_,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(source,Grid.piece_target()),source,offset),source,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))))
    Held.Both{ViewReads.whole_in(source,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),address_below),address_below} : ViewReads.InSource(Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component),source,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))))}
  source_cell_held(limits,source,source_room,source_positive,source_rank,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),within,found)

# Every source cell box index index of a target piece of a view reads lies in
# a piece its component took of the source.
def view_cell_held(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,
  +piece: U32,+length: U32,+shape_ok: Components.ShapeOk(limits,source),
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +length_le: {Nat.is_le(U32.to_nat(length),ShapeLaws.elements(source)) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(source)) == True{} : Bool},+fit: ViewReads.AxesFit(source,Grid.coordinates(source,offset),axes,Machine.half_word(limits)),
  +blocks: ViewReads.BlocksIn(Grid.grid(source,Grid.piece_target()),source,0,Machine.half_word(limits)),+separate: ViewSeparate(limits,source,axes,flags,offset,target,component),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),component),piece),
  +spans: Held.Spans,+laid: {spans == Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())) : Held.Spans},
  +index: Nat,+index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  Held.SpanHeld(spans,U32.to_nat(Part.cell_source(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),index,ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0))):
  Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = shape_ok
  +steps = ViewPart.view_steps(Views.mirrored_axes(axes,flags))
  +boxed = Box.grid_piece_box(limits,Views.axis_extents(axes),steps,target,room,nonempty,rank_room,piece,inside,index,index_in)
  +at_in = Box.box_end_of(Views.axis_extents(axes),steps,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),index,boxed)
  +cap_half = Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),LimitBounds.capacity_within_half(limits))
  +target_room = Order.transitive(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),room,cap_half)
  +length_room = Order.transitive(U32.to_nat(length),ShapeLaws.elements(source),U32.to_nat(Machine.half_word(limits)),length_le,
    Order.transitive(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),source_room,cap_half))
  +read = ViewReads.address_at(axes,flags,offset,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),length,Machine.half_word(limits),LimitBounds.half_word_fits(limits),target_room,length_room,fits,at_in)
  +address_below = Order.lt_le(Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),U32.to_nat(length),ShapeLaws.elements(source),read_below_of(offset,axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),length,read),length_le)
  +held = Held.by_cases(Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)))),Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),
    +yes => cell_separate(limits,source,axes,flags,offset,target,component,piece,source_room,source_positive,source_rank,room,nonempty,rank_room,offset_in,fit,blocks,yes,
      {%yes : Logic.When(_,Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>) separate : Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>},member,index,index_in,address_below,target_room,at_in),
    no => cell_whole(limits,source,axes,flags,offset,target,component,piece,index,source_room,source_positive,source_rank,no,address_below))
  %Equal.sym(U32,Part.cell_source(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),index,ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0),(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),ViewPart.view_piece_source(limits,axes,flags,offset,target,room,nonempty,rank_room,piece,inside,index,index_in)) :
    Held.SpanHeld(spans,U32.to_nat(_))
  %Equal.sym(Nat,U32.to_nat((Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32)),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))),read_exact_of(offset,axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index),(Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32),length,read)) : Held.SpanHeld(spans,_)
  %Equal.sym(Held.Spans,spans,Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())),laid) : Held.SpanHeld(_,Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index))))
  held

# Every box index from position on, remaining of them, of a target piece of a
# view reads a source cell its component took.
def view_cells_held(remaining: Nat,+position: Nat,+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +target: U32,+component: U32,+piece: U32,+length: U32,+shape_ok: Components.ShapeOk(limits,source),
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +length_le: {Nat.is_le(U32.to_nat(length),ShapeLaws.elements(source)) == True{} : Bool},
  +offset_in: {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(source)) == True{} : Bool},+fit: ViewReads.AxesFit(source,Grid.coordinates(source,offset),axes,Machine.half_word(limits)),
  +blocks: ViewReads.BlocksIn(Grid.grid(source,Grid.piece_target()),source,0,Machine.half_word(limits)),+separate: ViewSeparate(limits,source,axes,flags,offset,target,component),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),component),piece),
  +spans: Held.Spans,+laid: {spans == Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,component)),Grid.grid(source,Grid.piece_target())) : Held.Spans},
  +left: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  Part.CellsHeld(ViewPart.box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),ViewPart.view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0,spans,position,remaining):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{view_cell_held(limits,source,axes,flags,offset,target,component,piece,length,shape_ok,room,nonempty,rank_room,inside,fits,length_le,offset_in,fit,blocks,
          separate,member,spans,laid,position,Order.room_index(position,rest,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),left)),
        view_cells_held(rest,1n+position,limits,source,axes,flags,offset,target,component,piece,length,shape_ok,room,nonempty,rank_room,inside,fits,length_le,offset_in,
          fit,blocks,separate,member,spans,laid,Order.advance_room(position,rest,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),left))}

# ---- The pieces a component of a view took of its source ----

# ---- What a view gathers ----

# A target piece of a view holds the source values at the view addresses of
# its cells.
def ViewTarget(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,
  +first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Placed.PieceHolds(~Element,~Viewing<Logical>,~viewed(~Element,~Logical,~original),Viewing{logical,axes,offset,flags},first,rows,buffer)

# A view gathers a target piece as its model says, keeps the holdings in
# order and writes the piece with what the target property asks.
def view_gather_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: List<&2,U32>,+grid: Grid.Grid,+target: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,+holding: Held.Holding<Element>,
  +rest: Moved.Holdings<Element>,+model: PiecesModel.Model<Element>,+more: Moved.Models<Element>,
  +ordered: {Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},+later: Moved.HeldOrder(~Element,rest,more),
  +holds: ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags,first,count,
    Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.part(~Element,view_piece(Tensors.Viewed{source,grid,target,axes,offset,flags},first,count),Held.pack(~Element,holding),Tensors.buffer(~Element,fill,count))))))) ->
  Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
    Tensors.Viewed{source,grid,target,axes,offset,flags},components,first,count,fill,Moved.Holds{holding,rest},Moved.Models{model,more},
    view_model(~Element,Tensors.Viewed{source,grid,target,axes,offset,flags},components,first,count,fill,Moved.Holds{holding,rest})):
  +part = view_piece(Tensors.Viewed{source,grid,target,axes,offset,flags},first,count)
  +tree = Buffer.tree(~Element,fresh_output(~Element,fill,count))
  Held.Both{written_exact(~Element,part,holding,rest,fill,count),
    Held.Both{Held.Both{Equal.trans(Held.Stack<Element>,Held.order(~Element,held_after(~Element,part,holding,tree)),Held.order(~Element,holding),
          Held.stacked_model(~Element,model,Held.Bottom{}),Part.part_order(~Element,part_axes(part),part_source(part),part_target(part),holding,tree),ordered),later},
      holds}}

# ---- What every component of a separable view needs ----

# The classes of a component of a separable view cover the target pieces
# and end inside their blocks.
def view_classes_ok(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,
  +target_ok: Components.ShapeOk(limits,Views.axis_extents(axes)),+components_ok: Components.ComponentsOk([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Views.axis_extents(axes),Grid.grid(Views.axis_extents(axes),target),0),+rooms: Components.ReadingsRoom(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Machine.half_word(limits)),+ends: Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]))) == True{} : Bool}) ->
  Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits)):
  Held.Both{+room,Held.Both{+nonempty,+rank_room}} = target_ok
  +same = Equal.trans(Nat,List.length(&2,Tasks.Class,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),List.length(&2,Views.Axis,axes),
    Targets.chosen_rank(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],component),
    ViewReads.readings_length(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0))
  Targets.classes_ok(limits,Views.axis_extents(axes),target,room,nonempty,rank_room,[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],component,components_ok,limits_at,inside,
    ViewReads.view_class_ends(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0,
      Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Machine.half_word(limits),same,rooms,ends))

def separate_yes(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,
  +target_ok: Components.ShapeOk(limits,Views.axis_extents(axes)),+components_ok: Components.ComponentsOk([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Views.axis_extents(axes),Grid.grid(Views.axis_extents(axes),target),0),+separate: {Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]) == True{} : Bool},
  +row_ok: Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]))) == True{} : Bool}) ->
  ViewSeparate(limits,source,axes,flags,offset,target,component):
  Held.Both{+rooms,+more} = row_ok
  +cap = Machine.half_word(limits)
  +cap_ok = LimitBounds.half_word_fits(limits)
  +ends = Components.ends_at([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],0,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),cap,cap_ok,Components.runtime_ends([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),cap,cap_ok,components_ok,component,inside),{==},
    Components.components_rows([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),cap,components_ok))
  {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),True{},separate) : Logic.When(_,Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>)
    Held.Both{row_ok,Held.Both{ends,view_classes_ok(limits,source,axes,flags,offset,target,component,target_ok,components_ok,limits_at,rooms,ends,inside)}} :
    ViewSeparate(limits,source,axes,flags,offset,target,component)}

# Every component of a view below the class count has what a separable view
# needs of it.
def view_separate(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,+component: U32,
  +target_ok: Components.ShapeOk(limits,Views.axis_extents(axes)),+components_ok: Components.ComponentsOk([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Views.axis_extents(axes),Grid.grid(Views.axis_extents(axes),target),0),+row_ok: Logic.When(Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source)),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]))) == True{} : Bool}) ->
  ViewSeparate(limits,source,axes,flags,offset,target,component):
  Held.by_cases(ViewSeparate(limits,source,axes,flags,offset,target,component),Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),
    +yes => separate_yes(limits,source,axes,flags,offset,target,component,target_ok,components_ok,limits_at,yes,
      {%yes : Logic.When(_,Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source)) row_ok : Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source)},inside),
    +no => {%Equal.sym(Bool,Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),False{},no) : Logic.When(_,Held.Both<Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),component),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits))>>) Unit{} : ViewSeparate(limits,source,axes,flags,offset,target,component)})

# ---- The strides a target piece of a view steps by ----

# Every axis of a view that spans more than one index steps at most half a
# word.
def ViewStrides(axes: List<&2,Views.Axis>,+word: U32) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Views.Axis{+extent,+stride},rest}:
      Held.Both<Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}),ViewStrides(rest,word)>

def directed_twice(flag: Bool,+stride: U32) -> {Move.directed(flag,Views.directed(flag,stride)) == stride : U32}:
  match flag:
    case False{}: {==}
    case True{}: Arithmetic.negate_twice(stride)

def stride_one(+box: U32,+extent: U32,+stride: U32,+flag: Bool,+word: U32,
  +bounded: Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}),
  +wider: Logic.When(U32.is_lt(1,box),{U32.is_lt(1,extent) == True{} : Bool})) ->
  Logic.When(U32.is_lt(1,box),{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool}):
  Held.by_cases(Logic.When(U32.is_lt(1,box),{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool}),
    U32.is_lt(1,box),
    +wide => {%Equal.sym(Bool,U32.is_lt(1,box),True{},wide) : Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool})
      %Equal.sym(U32,Move.directed(flag,Views.directed(flag,stride)),stride,directed_twice(flag,stride)) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      %{%wide : Logic.When(_,{U32.is_lt(1,extent) == True{} : Bool}) wider : {U32.is_lt(1,extent) == True{} : Bool}} :
        Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool})
      bounded :
      Logic.When(U32.is_lt(1,box),{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool})},
    +narrow => {%Equal.sym(Bool,U32.is_lt(1,box),False{},narrow) :
        Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool})
      Unit{} :
      Logic.When(U32.is_lt(1,box),{Nat.is_le(Nat.double(U32.to_nat(Move.directed(flag,Views.directed(flag,stride)))),U32.to_nat(word)) == True{} : Bool})})

# The axes of a box of a view step at most half a word wherever they span
# more than one index.
def strides_boxed(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,boxes: List<&2,U32>,steps: List<&2,U32>,+word: U32,+views: ViewStrides(axes,word),
  +within: Box.BoxWithin(boxes,Views.axis_extents(axes))) ->
  Part.StridesRoom(Tensors.view_axes(ViewPart.boxed_axes(Views.mirrored_axes(axes,flags),boxes),flags,steps),word):
  match axes boxes steps:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later} Con{+step,+more}:
      Held.Both{+views_head,+views_rest} = views
      Held.Both{+within_head,+within_rest} = within
      Held.Both{stride_one(box,extent,stride,Views.flag(flags),word,views_head,Box.wider_one(box,extent,within_head)),
        strides_boxed(rest,Views.later_flags(flags),later,more,word,views_rest,within_rest)}
    case Nil{} +boxes +steps: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest} Nil{} +steps: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later} Nil{}: Unit{}

# ---- The law of the gather of a view ----

# A view that fits a source of length cells starts inside the source.
def offset_inside(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+length: U32,
  +source_ok: Components.ShapeOk(limits,source),+target_ok: Components.ShapeOk(limits,Views.axis_extents(axes)),
  +fits: {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool},
  +length_le: {Nat.is_le(U32.to_nat(length),ShapeLaws.elements(source)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(offset),ShapeLaws.elements(source)) == True{} : Bool}:
  Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = source_ok
  Held.Both{+room,Held.Both{+nonempty,+rank_room}} = target_ok
  +cap_half = Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),LimitBounds.capacity_within_half(limits))
  +target_room = Order.transitive(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),room,cap_half)
  +length_room = Order.transitive(U32.to_nat(length),ShapeLaws.elements(source),U32.to_nat(Machine.half_word(limits)),length_le,
    Order.transitive(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),source_room,cap_half))
  +first = (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,flags)),0n) : U32)
  +read = ViewReads.address_at(axes,flags,offset,0n,length,Machine.half_word(limits),LimitBounds.half_word_fits(limits),target_room,length_room,fits,nonempty)
  +sum = ViewReads.view_sum(axes,flags,0n)
  Order.le_lt_trans(U32.to_nat(offset),Nat.add(U32.to_nat(offset),sum),ShapeLaws.elements(source),Order.le_add(U32.to_nat(offset),sum),
    Order.lt_le(Nat.add(U32.to_nat(offset),sum),U32.to_nat(length),ShapeLaws.elements(source),read_below_of(offset,axes,flags,0n,first,length,read),length_le))

# A target piece of a view fits an allocation.
def span_allocatable(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool}) -> {U32.is_le(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),1073741824) == True{} : Bool}:
  Order.word_le(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Machine.capacity(limits),1073741824,
    U32Division.nat_le_native(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Machine.capacity(limits),Order.transitive(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),
      Box.grid_span_room(limits,Views.axis_extents(axes),target,room,nonempty,piece,inside),room)),
    LimitBounds.capacity_allocatable(limits))

# What a view of a source needs: the shapes fit the machine, the view fits
# the source, its axes, blocks and strides fit half a word, and its readings
# have class lists, blocks and rows that fit.
def ViewFacts(+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+target: U32) -> Data:
  Held.Both<Components.ShapeOk(limits,source),
  Held.Both<Components.ShapeOk(limits,Views.axis_extents(axes)),
  Held.Both<{Views.fits_nonempty(False{},Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool},
  Held.Both<{Nat.is_le(U32.to_nat(Tensors.count_or_zero(source)),ShapeLaws.elements(source)) == True{} : Bool},
  Held.Both<ViewReads.AxesFit(source,Grid.coordinates(source,offset),axes,Machine.half_word(limits)),
  Held.Both<ViewReads.BlocksIn(Grid.grid(source,Grid.piece_target()),source,0,Machine.half_word(limits)),
  Held.Both<ViewStrides(axes,Machine.half_word(limits)),
  Held.Both<Components.ComponentsOk([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Machine.half_word(limits)),
  Held.Both<{Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]))) == True{} : Bool},
  Held.Both<Components.LimitsAt([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Views.axis_extents(axes),Grid.grid(Views.axis_extents(axes),target),0),
    Logic.When(Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source))>>>>>>>>>>

# A gather context agrees with a view of a source: it reads the view into the
# grid of the view's extents, through the components of the view, from the
# one source, and the view fits.
def ViewAgrees(~Element: Data,+limits: Machine.Limits,+source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+context: Tensors.Viewed,
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+cells: List<&2,U32>,+cells_grid: Grid.Grid,+fill: Element) -> Data:
  Held.Both<{context == Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags} : Tensors.Viewed},
  Held.Both<{components == Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags) : Tasks.Components},
  Held.Both<{origins == [Tensors.origin(source,offset)] : List<&2,Tensors.Origin>},
  Held.Both<{cells == Views.axis_extents(axes) : List<&2,U32>},
  Held.Both<{cells_grid == Grid.grid(Views.axis_extents(axes),Grid.piece_target()) : Grid.Grid},
    ViewFacts(limits,source,axes,offset,flags,Grid.piece_target())>>>>>

def view_law_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+target: U32,+facts: ViewFacts(limits,source,axes,offset,flags,target),+fill: Element,+index: U32,
  +piece: U32,+holding: Held.Holding<Element>,+rest: Moved.Holdings<Element>,+model: PiecesModel.Model<Element>,+more: Moved.Models<Element>,
  +held: PiecesModel.ContentAll(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),PiecesModel.content(~Element,model)),
  +laid: {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.source_box(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),0,Grid.grid(source,Grid.piece_target()),source,offset,index)),Grid.grid(source,Grid.piece_target())) : Held.Spans},
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags)))) == True{} : Bool},
  +ordered: {Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},+later: Moved.HeldOrder(~Element,rest,more),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
    Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),fill,Moved.Holds{holding,rest},Moved.Models{model,more},
    view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),fill,Moved.Holds{holding,rest})):
  Held.Both{Held.Both{+source_room,Held.Both{+source_positive,+source_rank}},Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},
    Held.Both{+fits,Held.Both{+length_le,Held.Both{+fit,Held.Both{+blocks,Held.Both{+strides,Held.Both{+components_ok,
    Held.Both{+lists_positive,Held.Both{+limits_at,+row_ok}}}}}}}}}} = facts
  +source_ok = {Held.Both{source_room,Held.Both{source_positive,source_rank}} : Components.ShapeOk(limits,source)}
  +target_ok = {Held.Both{room,Held.Both{nonempty,rank_room}} : Components.ShapeOk(limits,Views.axis_extents(axes))}
  +component_in = Targets.component_lists(limits,[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],components_ok,lists_positive,index,inside)
  +separate = view_separate(limits,source,axes,flags,offset,target,index,target_ok,components_ok,limits_at,row_ok,component_in)
  +piece_in = Targets.target_piece_below(limits,Views.axis_extents(axes),target,room,nonempty,rank_room,[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)],index,Targets.when_third(~Components.RowOk(limits,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Grid.grid(source,Grid.piece_target()),source),~Components.RowEnds(Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),index)),
      ~Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),index),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits)),Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),separate),
    piece,member)
  +offset_in = offset_inside(limits,source,axes,flags,offset,Tensors.count_or_zero(source),source_ok,target_ok,fits,length_le)
  +stack = Held.stacked_model(~Element,model,Held.Bottom{})
  +span = U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))
  +cells = view_cells_held(span,0n,limits,source,axes,flags,offset,target,index,piece,Tensors.count_or_zero(source),source_ok,room,nonempty,rank_room,piece_in,fits,
    length_le,offset_in,fit,blocks,separate,member,Held.spans(~Element,stack),laid,Order.reflexive(span))
  +holds = view_target(~Element,~Logical,~original,logical,limits,source,axes,flags,offset,target,piece,holding,fill,room,nonempty,rank_room,piece_in,
    span_allocatable(limits,axes,target,room,nonempty,piece,piece_in),Machine.half_word(limits),LimitBounds.half_word_fits(limits),
    strides_boxed(axes,flags,Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),Tasks.strides_of(Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))),Machine.half_word(limits),strides,Box.grid_within(limits,Views.axis_extents(axes),target,room,nonempty,rank_room,piece,piece_in)),
    stack,ordered,0n,ShapeLaws.elements(source),source_room,
    Taken.row_sorted(~Element,~Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),index),Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Machine.half_word(limits)),
      limits,source,Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),[Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),index),offset,
      Tasks.separable([Tasks.readings(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Grid.coordinates(source,offset),axes,flags,0)]),model,source_ok,separate,
      PiecesModel.content_map(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),~Runs.piece_sized(~Element),~Placed.holds_sized(~Element,~Logical,~original,logical),
        PiecesModel.content(~Element,model),held),laid),
    Placed.ordered_values(~Element,~Logical,~original,logical,model,holding,ordered,held),cells)
  view_gather_facts(~Element,~Logical,~original,logical,source,Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),axes,offset,flags,Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),target),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),fill,holding,rest,model,more,ordered,later,holds)

def view_law_held(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+facts: ViewFacts(limits,source,axes,offset,flags,Grid.piece_target()),+fill: Element,+index: U32,
  +piece: U32,+models: Moved.Models<Element>,+holdings: Moved.Holdings<Element>,
  +ready: Moved.TakenReady(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical),Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),[Tensors.origin(source,offset)],models,index),
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags)))) == True{} : Bool},+order: Moved.HeldOrder(~Element,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
    Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,holdings,models,
    view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,holdings)):
  match models holdings:
    case Moved.Models{+model,+more} Moved.Holds{+holding,+rest}:
      Held.Both{Held.Both{+held,+held_rest},Held.Both{+laid,+laid_rest}} = ready
      Held.Both{+ordered,+later} = order
      view_law_at(~Element,~Logical,~original,logical,limits,source,axes,offset,flags,Grid.piece_target(),facts,fill,index,piece,holding,rest,model,more,held,laid,
        inside,ordered,later,member)
    case Moved.NoModels{} +holdings:
      Held.Both{+none,+laid} = ready
      Empty.absurd(Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
        Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,holdings,Moved.NoModels{},
        view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,holdings)),none)
    case Moved.Models{+model,+more} Moved.Unheld{}:
      Empty.absurd(Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
        Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,Moved.Unheld{},Moved.Models{model,more},
        view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),fill,Moved.Unheld{})),order)

# A view gathers every target piece of its components with the source values
# at the view addresses of its cells, as its model says, keeping the
# holdings in order.
def view_law(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+cells: List<&2,U32>,
  +cells_grid: Grid.Grid,+fill: Element,+context: Tensors.Viewed,+index: U32,+piece: U32,+models: Moved.Models<Element>,+holdings: Moved.Holdings<Element>,
  +agrees: ViewAgrees(~Element,limits,source,axes,offset,flags,context,components,origins,cells,cells_grid,fill),
  +ready: Moved.TakenReady(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical),components,origins,models,index),
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},+order: Moved.HeldOrder(~Element,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(components,cells_grid,cells,index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
    context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
    view_model(~Element,context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings)):
  Held.Both{+same_context,Held.Both{+same_components,Held.Both{+same_origins,Held.Both{+same_cells,Held.Both{+same_grid,+facts}}}}} = agrees
  %Equal.sym(Tensors.Viewed,context,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},same_context) :
    Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
      _,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
      view_model(~Element,_,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Tasks.Components,components,Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),same_components) :
    Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
      Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,
      view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Grid.Grid,cells_grid,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),same_grid) :
    Moved.GatherFacts(~Element,~Tensors.Viewed,~Tensors.view_gather(~Element),~ViewTarget(~Element,~Logical,~original,logical,axes,offset,flags),
      Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(_,piece),Grid.span(_,piece),fill,holdings,models,
      view_model(~Element,Tensors.Viewed{source,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),axes,offset,flags},Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.start(_,piece),Grid.span(_,piece),fill,holdings))
  view_law_held(~Element,~Logical,~original,logical,limits,source,axes,offset,flags,facts,fill,index,piece,models,holdings,
    {%same_origins : Moved.TakenReady(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical),Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),_,models,index)
      %same_components : Moved.TakenReady(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical),_,origins,models,index)
      ready : Moved.TakenReady(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical),Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),[Tensors.origin(source,offset)],models,index)},
    {%same_components : {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(_))) == True{} : Bool} inside :
      {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags)))) == True{} : Bool}},order,
    {%same_cells : Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),_,index),piece)
      %same_grid : Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),_,cells,index),piece)
      %same_components : Moved.Member(Tasks.target_pieces(_,cells_grid,cells,index),piece)
      member : Moved.Member(Tasks.target_pieces(Tasks.components(source,Grid.grid(source,Grid.piece_target()),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.View{axes,offset},flags),Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),index),piece)})
