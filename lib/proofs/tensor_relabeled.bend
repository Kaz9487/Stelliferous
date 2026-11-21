# Tensors.relabeled: the cells of a tensor under another shape of the same
# count hold the values they held, in the pieces of a grid of the same ranges
# or copied into the pieces of the grid of the new shape.
import Base
import ../tensor.bend as Tensors
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as Allocation
import ../traversal_partition.bend as Partition
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_move_readback.bend as Held
import ./tensor_joined.bend as Joined
import ./tensor_halo_windows.bend as HaloWindows
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./tensor_placed.bend as Placed
import ./tensor_task_runs.bend as Runs
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_shape_refinement.bend as ShapeLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as LimitBounds
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./nat_ceiling.bend as NatCeiling
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./nat_extrema.bend as Extrema
import ./word_encoding.bend as WordEncoding

# ---- Grids of the same ranges ----

def split_same(+left: Pieces.Ranges,+other_left: Pieces.Ranges,+right: Pieces.Ranges,+other_right: Pieces.Ranges,+lefts: {left == other_left : Pieces.Ranges},
  +rights: {right == other_right : Pieces.Ranges}) -> {Pieces.Split{left,right} == Pieces.Split{other_left,other_right} : Pieces.Ranges}:
  %lefts : {Pieces.Split{left,right} == Pieces.Split{_,other_right} : Pieces.Ranges}
  %rights : {Pieces.Split{left,right} == Pieces.Split{left,_} : Pieces.Ranges}
  {==}

# The ranges of a grid do not read its count of prefix axes.
def inner_same(depth: Nat,single: Bool,+count: U32,+prefix: U32,+other: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32,
  +first: U32,+rows: U32) ->
  {Pieces.inner_ranges(depth,single,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},first,rows) ==
    Pieces.inner_ranges(depth,single,Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},first,rows) : Pieces.Ranges}:
  match depth single:
    case 0n single: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +left = Pieces.below(rows)
      +right = (rows - left : U32)
      split_same(Pieces.inner_ranges(rest,U32.is_le(left,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},first,left),
        Pieces.inner_ranges(rest,U32.is_le(left,1),Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},first,left),
        Pieces.inner_ranges(rest,U32.is_le(right,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},(first + left : U32),right),
        Pieces.inner_ranges(rest,U32.is_le(right,1),Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},(first + left : U32),right),
        inner_same(rest,U32.is_le(left,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,first,left),
        inner_same(rest,U32.is_le(right,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,(first + left : U32),right))

def outer_same(depth: Nat,single: Bool,+count: U32,+prefix: U32,+other: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32,
  +step: U32,+first: U32,+rows: U32) ->
  {Pieces.outer_ranges(depth,single,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},step,first,rows) ==
    Pieces.outer_ranges(depth,single,Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},step,first,rows) : Pieces.Ranges}:
  match depth single:
    case 0n single: {==}
    case 1n+rest True{}: inner_same(32n,U32.is_le(step,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,(first * step : U32),step)
    case 1n+ +rest False{}:
      +left = Pieces.below(rows)
      +right = (rows - left : U32)
      split_same(Pieces.outer_ranges(rest,U32.is_le(left,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},step,first,left),
        Pieces.outer_ranges(rest,U32.is_le(left,1),Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},step,first,left),
        Pieces.outer_ranges(rest,U32.is_le(right,1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces},step,(first + left : U32),right),
        Pieces.outer_ranges(rest,U32.is_le(right,1),Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces},step,(first + left : U32),right),
        outer_same(rest,U32.is_le(left,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,step,first,left),
        outer_same(rest,U32.is_le(right,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,step,(first + left : U32),right))

def prefix_same(+count: U32,+prefix: U32,+other: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+pieces: U32) ->
  {Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,pieces}) ==
    Pieces.ranges_of_grid(Grid.Grid{count,other,outer,extent,unit_cells,block,per,pieces}) : Pieces.Ranges}:
  +groups = U32.max((outer / 1 : U32),1)
  outer_same(32n,U32.is_le(groups,1),count,prefix,other,outer,extent,unit_cells,block,per,pieces,(U32.max(per,1) * 1 : U32),0,groups)

# Units of the same outer indices, extent and unit cells make grids of the
# same ranges.
def units_same(+prefix: U32,+other: U32,+outer: U32,+other_outer: U32,+extent: U32,+other_extent: U32,+unit_cells: U32,+other_cells: U32,+target: U32,
  +outers: {outer == other_outer : U32},+extents: {extent == other_extent : U32},+cells: {unit_cells == other_cells : U32}) ->
  {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)) ==
    Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{other,other_outer,other_extent,other_cells},target)) : Pieces.Ranges}:
  %outers : {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)) ==
    Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{other,_,other_extent,other_cells},target)) : Pieces.Ranges}
  %extents : {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)) ==
    Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{other,outer,_,other_cells},target)) : Pieces.Ranges}
  %cells : {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)) ==
    Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{other,outer,extent,_},target)) : Pieces.Ranges}
  +block = Grid.piece_units(unit_cells,extent,target)
  +per = Grid.divided_up(extent,block)
  prefix_same((outer * extent * unit_cells : U32),prefix,other,outer,extent,unit_cells,block,per,(outer * per : U32))

def units_ranges(found: Grid.Units,other: Grid.Units,+target: U32,
  +same: {Grid.same_grid(Grid.grid_of_units(found,target),Grid.grid_of_units(other,target)) == True{} : Bool}) ->
  {Pieces.ranges_of_grid(Grid.grid_of_units(found,target)) == Pieces.ranges_of_grid(Grid.grid_of_units(other,target)) : Pieces.Ranges}:
  match found other:
    case Grid.Units{+prefix,+outer,+extent,+unit_cells} Grid.Units{+other_prefix,+other_outer,+other_extent,+other_cells}:
      +block = Grid.piece_units(unit_cells,extent,target)
      +other_block = Grid.piece_units(other_cells,other_extent,target)
      +counts = U32.is_eq((outer * extent * unit_cells : U32),(other_outer * other_extent * other_cells : U32))
      +outers = U32.is_eq(outer,other_outer)
      +extents = U32.is_eq(extent,other_extent)
      +cells = U32.is_eq(unit_cells,other_cells)
      +blocks = U32.is_eq(block,other_block)
      +four = Boolean.left(counts && outers && extents && cells,blocks,same)
      +three = Boolean.left(counts && outers && extents,cells,four)
      +two = Boolean.left(counts && outers,extents,three)
      units_same(prefix,other_prefix,outer,other_outer,extent,other_extent,unit_cells,other_cells,target,
        Arithmetic.equal_of_is_eq(outer,other_outer,Boolean.right(counts,outers,two)),
        Arithmetic.equal_of_is_eq(extent,other_extent,Boolean.right(counts && outers,extents,three)),
        Arithmetic.equal_of_is_eq(unit_cells,other_cells,Boolean.right(counts && outers && extents,cells,four)))

# Shapes of the same grid have the same ranges.
def same_ranges(+source: List<&2,U32>,+target: List<&2,U32>,+same: {Grid.same_grid(Tensors.grid_of(source),Tensors.grid_of(target)) == True{} : Bool}) ->
  {Pieces.ranges_of_grid(Tensors.grid_of(source)) == Pieces.ranges_of_grid(Tensors.grid_of(target)) : Pieces.Ranges}:
  +piece_target = Grid.piece_target()
  +left = GridLaws.grid_units(source,piece_target)
  +right = GridLaws.grid_units(target,piece_target)
  %Equal.sym(Grid.Grid,Grid.grid(source,piece_target),Grid.grid_of_units(Grid.units_from(source,Grid.strides(source),U32.is_le(Grid.product(source),piece_target),0,1,1,piece_target),piece_target),left) :
    {Pieces.ranges_of_grid(_) == Pieces.ranges_of_grid(Grid.grid(target,piece_target)) : Pieces.Ranges}
  %Equal.sym(Grid.Grid,Grid.grid(target,piece_target),Grid.grid_of_units(Grid.units_from(target,Grid.strides(target),U32.is_le(Grid.product(target),piece_target),0,1,1,piece_target),piece_target),right) :
    {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.units_from(source,Grid.strides(source),U32.is_le(Grid.product(source),piece_target),0,1,1,piece_target),piece_target)) == Pieces.ranges_of_grid(_) : Pieces.Ranges}
  units_ranges(Grid.units_from(source,Grid.strides(source),U32.is_le(Grid.product(source),piece_target),0,1,1,piece_target),Grid.units_from(target,Grid.strides(target),U32.is_le(Grid.product(target),piece_target),0,1,1,piece_target),piece_target,
    {%left : {Grid.same_grid(_,Grid.grid_of_units(Grid.units_from(target,Grid.strides(target),U32.is_le(Grid.product(target),piece_target),0,1,1,piece_target),piece_target)) == True{} : Bool}
      {%right : {Grid.same_grid(Grid.grid(source,piece_target),_) == True{} : Bool} same : {Grid.same_grid(Grid.grid(source,piece_target),Grid.grid_of_units(Grid.units_from(target,Grid.strides(target),U32.is_le(Grid.product(target),piece_target),0,1,1,piece_target),piece_target)) == True{} : Bool}} :
      {Grid.same_grid(Grid.grid_of_units(Grid.units_from(source,Grid.strides(source),U32.is_le(Grid.product(source),piece_target),0,1,1,piece_target),piece_target),Grid.grid_of_units(Grid.units_from(target,Grid.strides(target),U32.is_le(Grid.product(target),piece_target),0,1,1,piece_target),piece_target)) == True{} : Bool}})

# ---- The cells copied into the pieces of another grid ----

# The buffer a piece copies from the held pieces.
def copied_piece(~Element: Data,+holding: Held.Holding<Element>,+first: U32,+count: U32,+fill: Element) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))

# The held pieces after the copies of the ranges.
def copied_after(~Element: Data,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>) -> Held.Holding<Element>:
  match layout:
    case Pieces.Range{+first,+count}: Held.copied_holding(~Element,holding,first,count,fill)
    case Pieces.Split{left,right}: copied_after(~Element,right,fill,copied_after(~Element,left,fill,holding))

# The pieces the copies of the ranges make.
def copied_model(~Element: Data,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>) -> PiecesModel.Model<Element>:
  match layout:
    case Pieces.Range{+first,+count}: PiecesModel.Leaf{copied_piece(~Element,holding,first,count,fill),first,count}
    case Pieces.Split{+left,right}:
      PiecesModel.Fork{copied_model(~Element,left,fill,holding),copied_model(~Element,right,fill,copied_after(~Element,left,fill,holding))}

def copied_exact(~Element: Data,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>) ->
  {Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),layout,fill,Held.pack(~Element,holding)) ==
    (Held.pack(~Element,copied_after(~Element,layout,fill,holding)),PiecesModel.pack(~Element,copied_model(~Element,layout,fill,holding))) :
    Move.Holding<Element> & Pieces.Pieces(~Element)}:
  match layout:
    case Pieces.Range{+first,+count}:
      %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill),
          (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),
            Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill))),
          Held.pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill))) :
        {Tensors.flat_leaf(~Element,first,count,_) ==
          (Held.pack(~Element,Held.copied_holding(~Element,holding,first,count,fill)),Partition.Leaf{Buffer.pack(~Element,copied_piece(~Element,holding,first,count,fill)),first,count,Unit{}}) :
          Move.Holding<Element> & Pieces.Pieces(~Element)}
      %Held.copied_held(~Element,holding,first,count,fill) :
        {(Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),
            Partition.Leaf{Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),first,count,Unit{}}) ==
          (_,Partition.Leaf{Buffer.pack(~Element,copied_piece(~Element,holding,first,count,fill)),first,count,Unit{}}) : Move.Holding<Element> & Pieces.Pieces(~Element)}
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,copied_piece(~Element,holding,first,count,fill)),
          Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),
          Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))) :
        {(Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),
            Partition.Leaf{Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),first,count,Unit{}}) ==
          (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)),Partition.Leaf{_,first,count,Unit{}}) :
          Move.Holding<Element> & Pieces.Pieces(~Element)}
      {==}
    case Pieces.Split{+left,+right}:
      +left_after = copied_after(~Element,left,fill,holding)
      +left_model = copied_model(~Element,left,fill,holding)
      %Equal.sym(Move.Holding<Element> & Pieces.Pieces(~Element),
          Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),left,fill,Held.pack(~Element,holding)),
          (Held.pack(~Element,left_after),PiecesModel.pack(~Element,left_model)),copied_exact(~Element,left,fill,holding)) :
        {Storage.bind_pair(~Move.Holding<Element>,~Pieces.Pieces(~Element),~(Move.Holding<Element> & Pieces.Pieces(~Element)),_,
            held => built => Storage.bind_pair(~Move.Holding<Element>,~Pieces.Pieces(~Element),~(Move.Holding<Element> & Pieces.Pieces(~Element)),
              Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),right,fill,held),
              later => others => (later,Partition.Fork{built,others}))) ==
          (Held.pack(~Element,copied_after(~Element,right,fill,left_after)),
            Partition.Fork{PiecesModel.pack(~Element,left_model),PiecesModel.pack(~Element,copied_model(~Element,right,fill,left_after))}) :
          Move.Holding<Element> & Pieces.Pieces(~Element)}
      %Equal.sym(Move.Holding<Element> & Pieces.Pieces(~Element),
          Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),right,fill,Held.pack(~Element,left_after)),
          (Held.pack(~Element,copied_after(~Element,right,fill,left_after)),PiecesModel.pack(~Element,copied_model(~Element,right,fill,left_after))),
          copied_exact(~Element,right,fill,left_after)) :
        {Storage.bind_pair(~Move.Holding<Element>,~Pieces.Pieces(~Element),~(Move.Holding<Element> & Pieces.Pieces(~Element)),_,
            later => others => (later,Partition.Fork{PiecesModel.pack(~Element,left_model),others})) ==
          (Held.pack(~Element,copied_after(~Element,right,fill,left_after)),
            Partition.Fork{PiecesModel.pack(~Element,left_model),PiecesModel.pack(~Element,copied_model(~Element,right,fill,left_after))}) :
          Move.Holding<Element> & Pieces.Pieces(~Element)}
      {==}

def copied_laid(~Element: Data,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>) ->
  {PiecesModel.layout(~Element,copied_model(~Element,layout,fill,holding)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{+first,+count}: {==}
    case Pieces.Split{+left,+right}:
      split_same(PiecesModel.layout(~Element,copied_model(~Element,left,fill,holding)),left,
        PiecesModel.layout(~Element,copied_model(~Element,right,fill,copied_after(~Element,left,fill,holding))),right,
        copied_laid(~Element,left,fill,holding),copied_laid(~Element,right,fill,copied_after(~Element,left,fill,holding)))

# The held pieces stay laid over the cells and keep their values.
def copied_tiled(~Element: Data,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>,+total: Nat,+tiled: Held.Tiled(~Element,holding,0n,total)) ->
  Held.Tiled(~Element,copied_after(~Element,layout,fill,holding),0n,total):
  match layout:
    case Pieces.Range{+first,+count}:
      Held.run_tiled(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,
        Buffer.tree(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))),0n,total,tiled)
    case Pieces.Split{+left,+right}:
      copied_tiled(~Element,right,fill,copied_after(~Element,left,fill,holding),total,copied_tiled(~Element,left,fill,holding,total,tiled))

def copied_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,layout: Pieces.Ranges,+fill: Element,+holding: Held.Holding<Element>,
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,copied_after(~Element,layout,fill,holding)):
  match layout:
    case Pieces.Range{+first,+count}:
      Held.run_values(~Element,~Context,~value,context,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,
        Buffer.tree(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))),values)
    case Pieces.Split{+left,+right}:
      copied_values(~Element,~Context,~value,context,right,fill,copied_after(~Element,left,fill,holding),
        copied_values(~Element,~Context,~value,context,left,fill,holding,values))

# A piece copied from held pieces laid over the cells holds their values.
def copy_holds(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+limits: Machine.Limits,+holding: Held.Holding<Element>,+first: U32,+count: U32,
  +fill: Element,+total: Nat,+bounded: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+tiled: Held.Tiled(~Element,holding,0n,total),
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding),+within: PiecesPlan.LeafWithin<first,count,0n,total>) ->
  Placed.PieceHolds(~Element,~Context,~value,context,first,count,Buffer.content(~Element,copied_piece(~Element,holding,first,count,fill))):
  PiecesPlan.LeafWithin{+after,+fits} = within
  +capacity = Machine.capacity(limits)
  +count_room = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),Order.right_le(U32.to_nat(first),U32.to_nat(count)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),total,U32.to_nat(capacity),fits,bounded))
  +allocatable = Order.word_le(count,capacity,1073741824,
    {%Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)),Comparison.le(count,capacity)) : {_ == True{} : Bool}
      count_room : {U32.is_le(count,capacity) == True{} : Bool}},LimitBounds.capacity_allocatable(limits))
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill))),Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))) :
    Placed.PieceHolds(~Element,~Context,~value,context,first,count,_)
  +leaf = Held.copied_leaf(~Element,~Context,~value,context,holding,first,count,fill,0n,total,capacity,LimitBounds.capacity_fits(limits),bounded,tiled,values,after,fits,allocatable)
  KernelGemm.Both{
    Joined.content_values(~Element,~Context,~value,context,first,count,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill))),
      %Held.copy_tree(~Element,holding,first,count,fill) :
        Content.Leaf(~Element,~Context,~value,context,first,count,_,Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))))
      %Held.copy_length(~Element,holding,first,count,fill) :
        Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Held.pack(~Element,holding),first,count,fill)))),_)
      leaf),
    KernelGemm.Both{Held.copied_length(~Element,holding,first,count,fill),Held.copied_ready(~Element,holding,first,count,fill,allocatable)}}

def copied_holds(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+limits: Machine.Limits,layout: Pieces.Ranges,+fill: Element,
  +holding: Held.Holding<Element>,+total: Nat,+bounded: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+tiled: Held.Tiled(~Element,holding,0n,total),
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding),+within: PiecesPlan.Within(layout,0n,total)) ->
  PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Context,~value,context),PiecesModel.content(~Element,copied_model(~Element,layout,fill,holding))):
  match layout:
    case Pieces.Range{+first,+count}: copy_holds(~Element,~Context,~value,context,limits,holding,first,count,fill,total,bounded,tiled,values,within)
    case Pieces.Split{+left,+right}:
      KernelGemm.Both{+inside_left,+inside_right} = within
      KernelGemm.Both{copied_holds(~Element,~Context,~value,context,limits,left,fill,holding,total,bounded,tiled,values,inside_left),
        copied_holds(~Element,~Context,~value,context,limits,right,fill,copied_after(~Element,left,fill,holding),total,bounded,
          copied_tiled(~Element,left,fill,holding,total,tiled),copied_values(~Element,~Context,~value,context,left,fill,holding,values),inside_right)}

# ---- The contract under the new shape ----

# The grid ranges of a shape run from cell 0 to its last element.
def layout_coverage(+shape: List<&2,U32>,+layout: GridRanges.GridLayout<shape,Grid.piece_target(),1n>) ->
  PiecesPlan.Coverage(Pieces.ranges_of_grid(Tensors.grid_of(shape)),0n,ShapeLaws.elements(shape)):
  GridRanges.GridLayout{+run,+low,+high} = layout
  +ranges = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +total = ShapeLaws.elements(shape)
  +covered = Runs.run_coverage(ranges,1n,total,run)
  +covered_low = {%low : PiecesPlan.Coverage(ranges,_,Nat.sub(Runs.range_high(ranges),_)) covered : PiecesPlan.Coverage(ranges,0n,Nat.sub(Runs.range_high(ranges),0n))}
  +covered_high = {%high : PiecesPlan.Coverage(ranges,0n,Nat.sub(_,0n)) covered_low : PiecesPlan.Coverage(ranges,0n,Nat.sub(total,0n))}
  {%Order.subtract_zero(total) : PiecesPlan.Coverage(ranges,0n,_) covered_high : PiecesPlan.Coverage(ranges,0n,total)}

# Every piece of the grid ranges of a shape lies within its cells.
def layout_within(+shape: List<&2,U32>,+layout: GridRanges.GridLayout<shape,Grid.piece_target(),1n>) ->
  PiecesPlan.Within(Pieces.ranges_of_grid(Tensors.grid_of(shape)),0n,ShapeLaws.elements(shape)):
  +total = ShapeLaws.elements(shape)
  PiecesPlan.covered(Pieces.ranges_of_grid(Tensors.grid_of(shape)),0n,total,0n,total,layout_coverage(shape,layout),Order.reflexive(0n),Order.reflexive(total))

# The grid ranges of a counted shape of nonzero cells.
def target_layout(+limits: Machine.Limits,+target: List<&2,U32>,+count: U32,+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonzero: {U32.is_zero(count) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  GridRanges.GridLayout<target,Grid.piece_target(),1n>:
  +nonempty = {%Equal.sym(Nat,ShapeLaws.elements(target),U32.to_nat(count),Global.whole_counted(target,count,target_counted)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Arithmetic.word_nonzero_positive(count,nonzero) : {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}}
  GridRanges.grid_layout(limits,target,Grid.piece_target(),StorageCount.counted_room(limits,target,Global.Counted{count,target_counted},default_capacity),nonempty,1n,
    GridRanges.Multiple{GridRanges.last_extent(target),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(target),1n),GridRanges.last_extent(target),Order.times_one(GridRanges.last_extent(target)))})

# Held pieces of the source cells copied into the pieces of the grid of a
# shape of as many cells hold the values of the source.
def covered_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,-pieces: Pieces.Pieces(~Element),
  +source: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,+count: U32,+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +elements: {ShapeLaws.elements(source) == ShapeLaws.elements(target) : Nat},
  +bounded: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +layout: GridRanges.GridLayout<target,Grid.piece_target(),1n>,
  +covered: Held.Covered<Element,Logical,original,logical,Move.held(~Element,pieces),0n,ShapeLaws.elements(source),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(source)),Held.Ended{})>) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Tensors.flat_moved(~Element,Tensors.grid_of(target),fill,pieces),target,fill,Tensors.Usable{},workers}):
  Held.Covered{+holding,+packed,+tiled,+values,+ordered} = covered
  +ranges = Pieces.ranges_of_grid(Tensors.grid_of(target))
  +total = ShapeLaws.elements(source)
  +model = copied_model(~Element,ranges,fill,holding)
  %Equal.sym(Move.Holding<Element>,Move.held(~Element,pieces),Held.pack(~Element,holding),packed) :
    Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Pair.snd(Move.Holding<Element>,Pieces.Pieces(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),ranges,fill,_)),target,fill,Tensors.Usable{},workers})
  %Equal.sym(Move.Holding<Element> & Pieces.Pieces(~Element),
      Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~Tensors.flat_piece(~Element),ranges,fill,Held.pack(~Element,holding)),
      (Held.pack(~Element,copied_after(~Element,ranges,fill,holding)),PiecesModel.pack(~Element,model)),copied_exact(~Element,ranges,fill,holding)) :
    Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Pair.snd(Move.Holding<Element>,Pieces.Pieces(~Element),_),target,fill,Tensors.Usable{},workers})
  +within = {%Equal.sym(Nat,ShapeLaws.elements(source),ShapeLaws.elements(target),elements) : PiecesPlan.Within(ranges,0n,_)
    layout_within(target,layout) : PiecesPlan.Within(ranges,0n,total)}
  GridRanges.layout_contract(~Element,~Logical,~original,logical,target,fill,workers,PiecesModel.pack(~Element,model),1n,Global.Counted{count,target_counted},layout,
    Placed.PlacedModel{model,{==},copied_laid(~Element,ranges,fill,holding),
      copied_holds(~Element,~Logical,~original,logical,limits,ranges,fill,holding,total,bounded,tiled,values,within)})

def moved_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers}),
  +count: U32,+source_counted: {Tensors.count_of(source) == Some{count} : Maybe<&2,U32>},+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonzero: {U32.is_zero(count) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Contract(~Element,~Logical,~original,logical,
    Tensors.Tensor{Tensors.flat_moved(~Element,Tensors.grid_of(target),fill,PiecesModel.pack(~Element,model)),target,fill,Tensors.Usable{},workers}):
  +elements = Equal.trans(Nat,ShapeLaws.elements(source),U32.to_nat(count),ShapeLaws.elements(target),Global.whole_counted(source,count,source_counted),
    Equal.sym(Nat,ShapeLaws.elements(target),U32.to_nat(count),Global.whole_counted(target,count,target_counted)))
  covered_contract(~Element,~Logical,~original,logical,limits,PiecesModel.pack(~Element,model),source,target,fill,workers,count,target_counted,elements,
    StorageCount.counted_room(limits,source,Global.Counted{count,source_counted},default_capacity),
    target_layout(limits,target,count,target_counted,nonzero,default_capacity),
    HaloWindows.reduced_covered(~Element,~Logical,~original,logical,source,model,contract))

# Pieces of a grid of the same ranges are the pieces of the new shape.
def kept_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-pieces: Pieces.Pieces(~Element),+source: List<&2,U32>,
  +target: List<&2,U32>,+fill: Element,+workers: U32,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,source,fill,Tensors.Usable{},workers}),
  +count: U32,+source_counted: {Tensors.count_of(source) == Some{count} : Maybe<&2,U32>},+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +same: {Grid.same_grid(Tensors.grid_of(source),Tensors.grid_of(target)) == True{} : Bool}) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,target,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{+normal,+values} = contract
  KernelGemm.Both{+formed,+rest} = normal
  KernelGemm.Both{+counted,+canonical} = rest
  +elements = Equal.trans(Nat,ShapeLaws.elements(source),U32.to_nat(count),ShapeLaws.elements(target),Global.whole_counted(source,count,source_counted),
    Equal.sym(Nat,ShapeLaws.elements(target),U32.to_nat(count),Global.whole_counted(target,count,target_counted)))
  KernelGemm.Both{
    KernelGemm.Both{{%elements : Content.WellFormed<Element,_,PiecesModel.contents(~Element,pieces)> formed : Content.WellFormed<Element,ShapeLaws.elements(target),PiecesModel.contents(~Element,pieces)>},
      KernelGemm.Both{Global.Counted{count,target_counted},
        {%same_ranges(source,target,same) : {PiecesModel.content_layout(~Element,PiecesModel.contents(~Element,pieces)) == _ : Pieces.Ranges}
          canonical : Global.Canonical(~Element,target,PiecesModel.contents(~Element,pieces))}}},
    values}

# ---- Grids cut at the same places ----

# count ranges of size cells each from piece on, the last ending at the total.
def cut_spans(+total: Nat,+size: Nat,count: Nat,+piece: Nat) -> Held.Spans:
  match count:
    case 0n: Held.Ended{}
    case 1n+more:
      Held.Spanned{U32.from_nat(Nat.min(Nat.mul(piece,size),total)),
        U32.from_nat(Nat.sub(Nat.min(Nat.mul(1n+piece,size),total),Nat.min(Nat.mul(piece,size),total))),cut_spans(total,size,more,1n+piece)}

def spanned_same(+first: U32,+other_first: U32,+count: U32,+other_count: U32,+rest: Held.Spans,+other_rest: Held.Spans,
  +firsts: {first == other_first : U32},+counts: {count == other_count : U32},+rests: {rest == other_rest : Held.Spans}) ->
  {Held.Spanned{first,count,rest} == Held.Spanned{other_first,other_count,other_rest} : Held.Spans}:
  %firsts : {Held.Spanned{first,count,rest} == Held.Spanned{_,other_count,other_rest} : Held.Spans}
  %counts : {Held.Spanned{first,count,rest} == Held.Spanned{first,_,other_rest} : Held.Spans}
  %rests : {Held.Spanned{first,count,rest} == Held.Spanned{first,count,_} : Held.Spans}
  {==}

# A piece of a whole number of pieces per outer index starts at whole pieces
# of cells.
def exact_place(+piece: Nat,+per: Nat,+block: Nat,+unit: Nat,+extent: Nat,+exact: {extent == Nat.mul(per,block) : Nat},+positive: {Nat.is_lt(0n,per) == True{} : Bool}) ->
  {GridLaws.place(Nat.div(piece,per),Nat.mod(piece,per),Nat.mul(extent,unit),Nat.mul(block,unit)) == Nat.mul(piece,Nat.mul(block,unit)) : Nat}:
  +quotient = Nat.div(piece,per)
  +remainder = Nat.mod(piece,per)
  +size = Nat.mul(block,unit)
  +image = Equal.trans(Nat,Nat.mul(extent,unit),Nat.mul(Nat.mul(per,block),unit),Nat.mul(per,size),
    Equal.cong(Nat,Nat,value => Nat.mul(value,unit),extent,Nat.mul(per,block),exact),RowMajor.assoc(per,block,unit))
  +rows = Equal.trans(Nat,Nat.mul(quotient,Nat.mul(extent,unit)),Nat.mul(quotient,Nat.mul(per,size)),Nat.mul(Nat.mul(quotient,per),size),
    Equal.cong(Nat,Nat,value => Nat.mul(quotient,value),Nat.mul(extent,unit),Nat.mul(per,size),image),
    Equal.sym(Nat,Nat.mul(Nat.mul(quotient,per),size),Nat.mul(quotient,Nat.mul(per,size)),RowMajor.assoc(quotient,per,size)))
  Equal.trans(Nat,Nat.add(Nat.mul(quotient,Nat.mul(extent,unit)),Nat.mul(remainder,size)),Nat.add(Nat.mul(Nat.mul(quotient,per),size),Nat.mul(remainder,size)),Nat.mul(piece,size),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(remainder,size)),Nat.mul(quotient,Nat.mul(extent,unit)),Nat.mul(Nat.mul(quotient,per),size),rows),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.mul(quotient,per),size),Nat.mul(remainder,size)),Nat.mul(Nat.add(Nat.mul(quotient,per),remainder),size),Nat.mul(piece,size),
      Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,per),remainder),size),Nat.add(Nat.mul(Nat.mul(quotient,per),size),Nat.mul(remainder,size)),
        RowMajor.distrib(Nat.mul(quotient,per),remainder,size)),
      Equal.cong(Nat,Nat,value => Nat.mul(value,size),Nat.add(Nat.mul(quotient,per),remainder),piece,
        Equal.sym(Nat,piece,Nat.add(Nat.mul(quotient,per),remainder),DivisionOrder.quotient_exact(piece,per,positive)))))

# A piece of the one outer index starts at whole pieces of cells.
def single_place(+piece: Nat,+per: Nat,+image: Nat,+size: Nat,+below: {Nat.is_lt(piece,per) == True{} : Bool}) ->
  {GridLaws.place(Nat.div(piece,per),Nat.mod(piece,per),image,size) == Nat.mul(piece,size) : Nat}:
  %Equal.sym(Nat,Nat.div(piece,per),0n,DivisionOrder.small_quotient(piece,per,below)) : {GridLaws.place(_,Nat.mod(piece,per),image,size) == Nat.mul(piece,size) : Nat}
  %Equal.sym(Nat,Nat.mod(piece,per),piece,DivisionOrder.small_remainder(piece,per,below)) : {GridLaws.place(0n,_,image,size) == Nat.mul(piece,size) : Nat}
  {==}

# Every piece of a flat grid below its end starts at whole pieces of cells.
def flat_place(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {GridLaws.place(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.mul(piece,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat}:
  match first:
    case True{}:
      GridLaws.Shape{+outer_positive,unit_positive,block_positive,block_below,per_exact,room} = shape
      +single = Order.antisymmetric(U32.to_nat(outer),1n,
        Equal.trans(Bool,Nat.is_le(U32.to_nat(outer),1n),U32.is_le(outer,1),True{},Equal.sym(Bool,U32.is_le(outer,1),Nat.is_le(U32.to_nat(outer),1n),Comparison.le(outer,1)),first_is),
        Order.positive_one(U32.to_nat(outer),outer_positive))
      +one_times = Equal.trans(Nat,Nat.mul(1n,U32.to_nat(per)),Nat.mul(U32.to_nat(per),1n),U32.to_nat(per),RowMajor.multiply_commutative(1n,U32.to_nat(per)),Order.times_one(U32.to_nat(per)))
      +below = {%one_times : {Nat.is_lt(piece,_) == True{} : Bool}
        {%single : {Nat.is_lt(piece,Nat.mul(_,U32.to_nat(per))) == True{} : Bool} inside : {Nat.is_lt(piece,Nat.mul(1n,U32.to_nat(per))) == True{} : Bool}} : {Nat.is_lt(piece,U32.to_nat(per)) == True{} : Bool}}
      single_place(piece,U32.to_nat(per),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),below)
    case False{}:
      match second:
        case True{}:
          GridLaws.Shape{outer_positive,+unit_positive,+block_positive,block_below,+per_exact,room} = shape
          +block_room = Order.transitive(U32.to_nat(block),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(block),U32.to_nat(unit_cells),unit_positive),
            GridLaws.cells_room(limits,outer,extent,unit_cells,block,per,shape))
          +kept = GridLaws.maximum_kept(block,1,Order.positive_one(U32.to_nat(block),block_positive))
          +remainder_zero = Equal.trans(Nat,Nat.mod(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat((extent % block : U32)),0n,
            Equal.sym(Nat,U32.to_nat((extent % block : U32)),Nat.mod(U32.to_nat(extent),U32.to_nat(block)),GridLaws.word_remainder(limits,extent,block,block_room,block_positive)),
            {%kept : {U32.to_nat((extent % _ : U32)) == 0n : Nat}
              Equal.cong(U32,Nat,value => U32.to_nat(value),(extent % U32.max(block,1) : U32),0,Arithmetic.word_zero((extent % U32.max(block,1) : U32),second_is)) :
              {U32.to_nat((extent % block : U32)) == 0n : Nat}})
          +per_quotient = Equal.trans(Nat,U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),Nat.div(U32.to_nat(extent),U32.to_nat(block)),per_exact,
            Equal.cong(Nat,Nat,value => NatCeiling.rounded(Nat.div(U32.to_nat(extent),U32.to_nat(block)),value),Nat.mod(U32.to_nat(extent),U32.to_nat(block)),0n,remainder_zero))
          +whole = Nat.mul(Nat.div(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(block))
          +exact = Equal.trans(Nat,U32.to_nat(extent),Nat.add(whole,Nat.mod(U32.to_nat(extent),U32.to_nat(block))),Nat.mul(U32.to_nat(per),U32.to_nat(block)),DivisionOrder.quotient_exact(U32.to_nat(extent),U32.to_nat(block),block_positive),
            Equal.trans(Nat,Nat.add(whole,Nat.mod(U32.to_nat(extent),U32.to_nat(block))),Nat.add(whole,0n),Nat.mul(U32.to_nat(per),U32.to_nat(block)),
              Equal.cong(Nat,Nat,value => Nat.add(whole,value),Nat.mod(U32.to_nat(extent),U32.to_nat(block)),0n,remainder_zero),
              Equal.trans(Nat,Nat.add(whole,0n),whole,Nat.mul(U32.to_nat(per),U32.to_nat(block)),Addition.zero_right(whole),
                Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),Nat.div(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(per),Equal.sym(Nat,U32.to_nat(per),Nat.div(U32.to_nat(extent),U32.to_nat(block)),per_quotient)))))
          exact_place(piece,U32.to_nat(per),U32.to_nat(block),U32.to_nat(unit_cells),U32.to_nat(extent),exact,GridLaws.per_positive(limits,outer,extent,unit_cells,block,per,shape))
        case False{}:
          Empty.absurd({GridLaws.place(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) == Nat.mul(piece,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))) : Nat},WordEncoding.false_true(flat))

# The cells of a piece of a grid in a word.
def piece_size(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>) -> {U32.to_nat((unit_cells * block : U32)) == Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)) : Nat}:
  Equal.trans(Nat,U32.to_nat((unit_cells * block : U32)),U32.to_nat((block * unit_cells : U32)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),
    Equal.cong(U32,Nat,value => U32.to_nat(value),(unit_cells * block : U32),(block * unit_cells : U32),Arithmetic.multiply_commutative(unit_cells,block)),
    GridLaws.cells_word(limits,outer,extent,unit_cells,block,per,shape))

def size_positive(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_lt(0n,U32.to_nat((unit_cells * block : U32))) == True{} : Bool}:
  GridLaws.Shape{outer_positive,+unit_positive,+block_positive,block_below,per_exact,room} = shape
  {%Equal.sym(Nat,U32.to_nat((unit_cells * block : U32)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),piece_size(limits,outer,extent,unit_cells,block,per,count,shape)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.lt_le(0n,U32.to_nat(block),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),block_positive,RowMajor.product_covers(U32.to_nat(block),U32.to_nat(unit_cells),unit_positive)) : {Nat.is_lt(0n,U32.to_nat((unit_cells * block : U32))) == True{} : Bool}}

def count_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) -> {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  GridLaws.Shape{outer_positive,unit_positive,block_positive,block_below,per_exact,+room} = shape
  Held.nat_le_moved(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),cells,room)

# A piece below the end in a word.
def piece_word(+limits: Machine.Limits,+outer: U32,+per: U32,+piece: Nat,+inside: {Nat.is_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(piece)) == piece : Nat}:
  Bounds.roundtrip(piece,Machine.capacity(limits),LimitBounds.capacity_fits(limits),Order.transitive(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),inside,room))

def piece_inside(+limits: Machine.Limits,+outer: U32,+per: U32,+piece: Nat,+inside: {Nat.is_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(U32.from_nat(piece)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}:
  {%Equal.sym(Nat,U32.to_nat(U32.from_nat(piece)),piece,piece_word(limits,outer,per,piece,inside,room)) : {Nat.is_le(_,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}
    inside : {Nat.is_le(U32.to_nat(U32.from_nat(piece)),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}

def below_le(+piece: Nat,+end: Nat,+below: {Nat.is_lt(piece,end) == True{} : Bool}) -> {Nat.is_le(piece,end) == True{} : Bool}:
  Order.transitive(piece,1n+piece,end,Order.successor(piece,piece,Order.reflexive(piece)),Order.strict_follow(piece,end,below))

# A flat grid starts every piece below its end at whole pieces of cells.
def start_flat(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))) == Nat.mul(piece,U32.to_nat((unit_cells * block : U32))) : Nat}:
  +room = GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
  +word = piece_word(limits,outer,per,piece,below_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside),room)
  +exact = U32.to_nat(U32.from_nat(piece))
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))),GridLaws.place(Nat.div(exact,U32.to_nat(per)),Nat.mod(exact,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),
    GridLaws.piece_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(piece),piece_inside(limits,outer,per,piece,below_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside),room)),
    Equal.trans(Nat,GridLaws.place(Nat.div(exact,U32.to_nat(per)),Nat.mod(exact,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),GridLaws.place(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),
      Equal.cong(Nat,Nat,value => GridLaws.place(Nat.div(value,U32.to_nat(per)),Nat.mod(value,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),exact,piece,word),
      Equal.trans(Nat,GridLaws.place(Nat.div(piece,U32.to_nat(per)),Nat.mod(piece,U32.to_nat(per)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(piece,Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),
        flat_place(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,first_is,second_is,flat,piece,inside),
        Equal.cong(Nat,Nat,value => Nat.mul(piece,value),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),U32.to_nat((unit_cells * block : U32)),Equal.sym(Nat,U32.to_nat((unit_cells * block : U32)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),piece_size(limits,outer,extent,unit_cells,block,per,count,shape))))))

# Every piece of a flat grid below its end starts at the least of whole
# pieces of cells and the cells.
def start_cut(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},+piece: Nat,+inside: {Nat.is_lt(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece))) == Nat.min(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)) : Nat}:
  +start = start_flat(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,first_is,second_is,flat,piece,inside)
  +value = U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(piece)))
  +before = GridRanges.start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(piece),
    piece_inside(limits,outer,per,piece,below_le(piece,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),inside),GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)))
  +ordered = Held.nat_le_moved(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),value,U32.to_nat(count),Equal.sym(Nat,value,Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),start),before)
  Equal.trans(Nat,value,Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),Nat.min(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),start,
    Equal.sym(Nat,Nat.min(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),Extrema.minimum_when_ordered(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count),ordered)))

# The pieces of a grid cover its cells.
def cells_covered(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) -> {Nat.is_le(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32)))) == True{} : Bool}:
  GridLaws.Shape{outer_positive,unit_positive,+block_positive,block_below,+per_exact,room} = shape
  +rows = Nat.mul(U32.to_nat(per),U32.to_nat(block))
  +extent_covered = Held.nat_le_right_moved(U32.to_nat(extent),rows,Nat.mul(NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),U32.to_nat(block)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat(per),NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block)),per_exact),NatCeiling.covers(U32.to_nat(extent),U32.to_nat(block),block_positive))
  +scaled = RowMajor.mul_le(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(U32.to_nat(outer),rows),U32.to_nat(unit_cells),RowMajor.mul_le_left(U32.to_nat(outer),U32.to_nat(extent),rows,extent_covered))
  +regrouped = Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),rows),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(block)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),rows),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(block)),Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(block)),Nat.mul(U32.to_nat(outer),rows),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(per),U32.to_nat(block)))),
    RowMajor.assoc(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(block),U32.to_nat(unit_cells)))
  +sized = Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),value),U32.to_nat((unit_cells * block : U32)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells)),piece_size(limits,outer,extent,unit_cells,block,per,count,shape))
  Held.nat_le_moved(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),cells,
    Held.nat_le_right_moved(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),sized,
      Held.nat_le_right_moved(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),rows),U32.to_nat(unit_cells)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),rows),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.mul(U32.to_nat(block),U32.to_nat(unit_cells))),regrouped),scaled)))

# The end of a grid starts at its cells.
def end_cells(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) -> {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))) == U32.to_nat(count) : Nat}:
  +pieces_word = GridLaws.pieces_word(limits,outer,extent,unit_cells,block,per,shape)
  +word = piece_word(limits,outer,per,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +same = Arithmetic.to_nat_injective(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),(outer * per : U32),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((outer * per : U32)),word,Equal.sym(Nat,U32.to_nat((outer * per : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),pieces_word)))
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),U32.to_nat(count),
    Equal.cong(U32,Nat,piece => U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),(outer * per : U32),same),
    Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(outer * per : U32))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(count),GridRanges.end_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape),
      Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),cells)))

# The end of a grid starts at the least of whole pieces of cells and the cells.
def end_cut(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat}) -> {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))) == Nat.min(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)) : Nat}:
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))),U32.to_nat(count),Nat.min(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),end_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells),
    Equal.sym(Nat,Nat.min(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),U32.to_nat(count),Equal.trans(Nat,Nat.min(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),Nat.min(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32)))),U32.to_nat(count),
      Extrema.minimum_commutative(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),Extrema.minimum_when_ordered(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat((unit_cells * block : U32))),cells_covered(limits,outer,extent,unit_cells,block,per,count,shape,cells)))))

# A piece with more pieces after it is below their end.
def later_below(+piece: Nat,+more: Nat,+end: Nat,+reach: {Nat.add(piece,1n+more) == end : Nat}) -> {Nat.is_lt(piece,end) == True{} : Bool}:
  %reach : {Nat.is_lt(piece,_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)) : {Nat.is_lt(piece,_) == True{} : Bool}
  Order.lt_succ(piece,Nat.add(piece,more),Order.le_add(piece,more))

# The piece after one with more pieces after it starts at the least of whole
# pieces of cells and the cells.
def next_cut(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},more: Nat,+piece: Nat,+reach: {Nat.add(piece,1n+more) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece))) == Nat.min(Nat.mul(1n+piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)) : Nat}:
  match more:
    case 0n:
      +last = Equal.trans(Nat,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Nat.add(piece,1n),1n+piece,Equal.sym(Nat,Nat.add(piece,1n),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),reach),
        Equal.trans(Nat,Nat.add(piece,1n),1n+Nat.add(piece,0n),1n+piece,Addition.successor_right(piece,0n),
          Equal.cong(Nat,Nat,value => 1n+value,Nat.add(piece,0n),piece,Addition.zero_right(piece))))
      %last : {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(_))) == Nat.min(Nat.mul(_,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)) : Nat}
      end_cut(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells)
    case 1n+ +rest:
      start_cut(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first_is,second_is,flat,1n+piece,
        later_below(1n+piece,rest,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Equal.trans(Nat,1n+Nat.add(piece,1n+rest),Nat.add(piece,1n+(1n+rest)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
          Equal.sym(Nat,Nat.add(piece,1n+(1n+rest)),1n+Nat.add(piece,1n+rest),Addition.successor_right(piece,1n+rest)),reach)))

# The ranges of the pieces of a flat grid from piece on are cuts of its cells.
def flat_cuts(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},left: Nat,+piece: Nat,+reach: {Nat.add(piece,left) == Nat.mul(U32.to_nat(outer),U32.to_nat(per)) : Nat}) ->
  {GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},left,piece,Held.Ended{}) == cut_spans(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),left,piece) : Held.Spans}:
  match left:
    case 0n: {==}
    case 1n+ +more:
      +cap = Machine.capacity(limits)
      +below = later_below(piece,more,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),reach)
      +here = U32.from_nat(piece)
      +low = Nat.min(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count))
      +high = Nat.min(Nat.mul(1n+piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count))
      +start = U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here))
      +span = U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here,1))
      +next = U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+piece)))
      +starts = start_cut(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first_is,second_is,flat,piece,below)
      +nexts = next_cut(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first_is,second_is,flat,more,piece,reach)
      +span_value = Equal.trans(Nat,span,Nat.sub(next,start),Nat.sub(high,low),
        Equal.trans(Nat,span,Nat.sub(Nat.add(start,span),start),Nat.sub(next,start),
          Equal.sym(Nat,Nat.sub(Nat.add(start,span),start),span,Order.subtract_shift(start,span)),
          Equal.cong(Nat,Nat,value => Nat.sub(value,start),Nat.add(start,span),next,GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,below))),
        Equal.trans(Nat,Nat.sub(next,start),Nat.sub(high,start),Nat.sub(high,low),
          Equal.cong(Nat,Nat,value => Nat.sub(value,start),next,high,nexts),
          Equal.cong(Nat,Nat,value => Nat.sub(high,value),start,low,starts)))
      +room = count_room(limits,outer,extent,unit_cells,block,per,count,shape,cells)
      +low_room = Order.transitive(low,U32.to_nat(count),U32.to_nat(cap),Extrema.minimum_right(Nat.mul(piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),room)
      +span_room = Order.transitive(Nat.sub(high,low),high,U32.to_nat(cap),Order.sub_le(high,low),
        Order.transitive(high,U32.to_nat(count),U32.to_nat(cap),Extrema.minimum_right(Nat.mul(1n+piece,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)),room))
      +first_word = Arithmetic.to_nat_injective(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here),U32.from_nat(low),
        Equal.trans(Nat,start,low,U32.to_nat(U32.from_nat(low)),starts,
          Equal.sym(Nat,U32.to_nat(U32.from_nat(low)),low,Bounds.roundtrip(low,cap,LimitBounds.capacity_fits(limits),low_room))))
      +count_word = Arithmetic.to_nat_injective(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here,1),U32.from_nat(Nat.sub(high,low)),
        Equal.trans(Nat,span,Nat.sub(high,low),U32.to_nat(U32.from_nat(Nat.sub(high,low))),span_value,
          Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.sub(high,low))),Nat.sub(high,low),Bounds.roundtrip(Nat.sub(high,low),cap,LimitBounds.capacity_fits(limits),span_room))))
      +reach_next = Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
        Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),reach)
      spanned_same(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here),U32.from_nat(low),Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},here,1),U32.from_nat(Nat.sub(high,low)),
        GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},more,1n+piece,Held.Ended{}),cut_spans(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),more,1n+piece),first_word,count_word,
        flat_cuts(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first_is,second_is,flat,more,1n+piece,reach_next))

# Pieces of a flat grid: no fewer cover its cells.
def pieces_least(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool},pieces: Nat,+exact: {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == pieces : Nat}) ->
  {Nat.is_le(pieces,NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)))) == True{} : Bool}:
  match pieces:
    case 0n: Order.zero_le(NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))))
    case 1n+ +last:
      +room = GridLaws.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
      +below = {%Equal.sym(Nat,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),1n+last,exact) : {Nat.is_lt(last,_) == True{} : Bool}
        Order.lt_succ(last,last,Order.reflexive(last)) : {Nat.is_lt(last,Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}}
      +last_word = piece_word(limits,outer,per,last,below_le(last,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),below),room)
      +end_word = piece_word(limits,outer,per,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),room)
      +before = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(last)),last,last_word) : {Nat.is_lt(_,U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))) == True{} : Bool}
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),end_word) : {Nat.is_lt(last,_) == True{} : Bool}
          below : {Nat.is_lt(last,U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))) == True{} : Bool}} :
        {Nat.is_lt(U32.to_nat(U32.from_nat(last)),U32.to_nat(U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))))) == True{} : Bool}}
      +ordered = GridLaws.start_order(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(last),U32.from_nat(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),
        piece_inside(limits,outer,per,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),Order.reflexive(Nat.mul(U32.to_nat(outer),U32.to_nat(per))),room),before)
      +short = {%start_flat(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,first_is,second_is,flat,last,below) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
        {%end_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells) : {Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(last))),_) == True{} : Bool}
          ordered : {Nat.is_lt(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(last))),U32.to_nat(count)) == True{} : Bool}} :
        {Nat.is_lt(Nat.mul(last,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count)) == True{} : Bool}}
      Order.strict_follow(last,NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),
        RowMajor.cancel_multiply_lt(last,NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),U32.to_nat((unit_cells * block : U32)),
          Order.lt_le(Nat.mul(last,U32.to_nat((unit_cells * block : U32))),U32.to_nat(count),Nat.mul(NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),U32.to_nat((unit_cells * block : U32))),short,NatCeiling.covers(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),size_positive(limits,outer,extent,unit_cells,block,per,count,shape)))))

# A flat grid has the least pieces that cover its cells.
def pieces_ceiling(+first: Bool,+second: Bool,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,+shape: GridLaws.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+first_is: {U32.is_le(outer,1) == first : Bool},+second_is: {U32.is_zero((extent % U32.max(block,1) : U32)) == second : Bool},+flat: {first || second == True{} : Bool}) -> {Nat.mul(U32.to_nat(outer),U32.to_nat(per)) == NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))) : Nat}:
  Order.antisymmetric(Nat.mul(U32.to_nat(outer),U32.to_nat(per)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),pieces_least(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,first_is,second_is,flat,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),{==}),
    NatCeiling.minimal(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),size_positive(limits,outer,extent,unit_cells,block,per,count,shape),cells_covered(limits,outer,extent,unit_cells,block,per,count,shape,cells)))

# The ranges of a flat grid are the cuts of its cells.
def grid_cuts(+limits: Machine.Limits,+grid: Grid.Grid,+facts: GridLaws.GridFacts<limits,grid>,+flat: {Grid.flat_blocks(grid) == True{} : Bool}) ->
  {Held.layout_spans(Pieces.ranges_of_grid(grid),Held.Ended{}) ==
    cut_spans(U32.to_nat(Grid.count_of(grid)),U32.to_nat(Grid.piece_cells(grid)),NatCeiling.ceiling(U32.to_nat(Grid.count_of(grid)),U32.to_nat(Grid.piece_cells(grid))),0n) : Held.Spans}:
  GridLaws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  +flat_grid = {%exact : {Grid.flat_blocks(_) == True{} : Bool} flat : {Grid.flat_blocks(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}) == True{} : Bool}}
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact) :
    {Held.layout_spans(Pieces.ranges_of_grid(_),Held.Ended{}) ==
      cut_spans(U32.to_nat(Grid.count_of(_)),U32.to_nat(Grid.piece_cells(_)),NatCeiling.ceiling(U32.to_nat(Grid.count_of(_)),U32.to_nat(Grid.piece_cells(_))),0n) : Held.Spans}
  +first = U32.is_le(outer,1)
  +second = U32.is_zero((extent % U32.max(block,1) : U32))
  Equal.trans(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
    cut_spans(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),0n),
    GridRanges.grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape),
    {%pieces_ceiling(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,{==},{==},flat_grid) : {GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}) == cut_spans(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),_,0n) : Held.Spans}
      flat_cuts(first,second,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,{==},{==},flat_grid,Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,{==}) : {GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}) == cut_spans(U32.to_nat(count),U32.to_nat((unit_cells * block : U32)),NatCeiling.ceiling(U32.to_nat(count),U32.to_nat((unit_cells * block : U32))),0n) : Held.Spans}})

# Two shapes whose grids cut their cells at the same places have pieces of the
# same ranges in order.
def same_cuts(+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +same: {Grid.same_pieces(Tensors.grid_of(source),Tensors.grid_of(target)) == True{} : Bool}) ->
  {Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(source)),Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(target)),Held.Ended{}) : Held.Spans}:
  +left = Tensors.grid_of(source)
  +right = Tensors.grid_of(target)
  +counts_flat = Boolean.and_true_left(U32.is_eq(Grid.count_of(left),Grid.count_of(right)) && Grid.flat_blocks(left) && Grid.flat_blocks(right),
    U32.is_eq(Grid.piece_cells(left),Grid.piece_cells(right)),same)
  +counts_left = Boolean.and_true_left(U32.is_eq(Grid.count_of(left),Grid.count_of(right)) && Grid.flat_blocks(left),Grid.flat_blocks(right),counts_flat)
  +counts = Arithmetic.equal_of_is_eq(Grid.count_of(left),Grid.count_of(right),
    Boolean.and_true_left(U32.is_eq(Grid.count_of(left),Grid.count_of(right)),Grid.flat_blocks(left),counts_left))
  +left_flat = Boolean.and_true_right(U32.is_eq(Grid.count_of(left),Grid.count_of(right)),Grid.flat_blocks(left),counts_left)
  +right_flat = Boolean.and_true_right(U32.is_eq(Grid.count_of(left),Grid.count_of(right)) && Grid.flat_blocks(left),Grid.flat_blocks(right),counts_flat)
  +sizes = Arithmetic.equal_of_is_eq(Grid.piece_cells(left),Grid.piece_cells(right),
    Boolean.and_true_right(U32.is_eq(Grid.count_of(left),Grid.count_of(right)) && Grid.flat_blocks(left) && Grid.flat_blocks(right),
      U32.is_eq(Grid.piece_cells(left),Grid.piece_cells(right)),same))
  +left_size = U32.to_nat(Grid.piece_cells(left))
  +right_count = U32.to_nat(Grid.count_of(right))
  Equal.trans(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(left),Held.Ended{}),
    cut_spans(U32.to_nat(Grid.count_of(left)),left_size,NatCeiling.ceiling(U32.to_nat(Grid.count_of(left)),left_size),0n),
    Held.layout_spans(Pieces.ranges_of_grid(right),Held.Ended{}),
    grid_cuts(limits,left,GridRanges.grid_facts(limits,source,Grid.piece_target(),source_room,source_nonempty),left_flat),
    Equal.trans(Held.Spans,cut_spans(U32.to_nat(Grid.count_of(left)),left_size,NatCeiling.ceiling(U32.to_nat(Grid.count_of(left)),left_size),0n),
      cut_spans(right_count,left_size,NatCeiling.ceiling(right_count,left_size),0n),
      Held.layout_spans(Pieces.ranges_of_grid(right),Held.Ended{}),
      Equal.cong(U32,Held.Spans,cells => cut_spans(U32.to_nat(cells),left_size,NatCeiling.ceiling(U32.to_nat(cells),left_size),0n),Grid.count_of(left),Grid.count_of(right),counts),
      Equal.trans(Held.Spans,cut_spans(right_count,left_size,NatCeiling.ceiling(right_count,left_size),0n),
        cut_spans(right_count,U32.to_nat(Grid.piece_cells(right)),NatCeiling.ceiling(right_count,U32.to_nat(Grid.piece_cells(right))),0n),
        Held.layout_spans(Pieces.ranges_of_grid(right),Held.Ended{}),
        Equal.cong(U32,Held.Spans,size => cut_spans(right_count,U32.to_nat(size),NatCeiling.ceiling(right_count,U32.to_nat(size)),0n),Grid.piece_cells(left),Grid.piece_cells(right),sizes),
        Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(right),Held.Ended{}),
          cut_spans(right_count,U32.to_nat(Grid.piece_cells(right)),NatCeiling.ceiling(right_count,U32.to_nat(Grid.piece_cells(right))),0n),
          grid_cuts(limits,right,GridRanges.grid_facts(limits,target,Grid.piece_target(),target_room,target_nonempty),right_flat)))))

# ---- Pieces regrouped by other ranges ----

# The buffers of pieces in order.
type Buffers<-Element: Data> is Data:
  Unbuffered{}
  Buffered{buffer: Buffer.Content<Element>,rest: Buffers<Element>}

# Pieces in order, each with its range.
type Line<-Element: Data> is Data:
  Unlined{}
  Lined{buffer: Buffer.Content<Element>,first: U32,rows: U32,rest: Line<Element>}

def content_buffers(~Element: Data,content: PiecesModel.Content<Element>,rest: Buffers<Element>) -> Buffers<Element>:
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}: Buffered{buffer,rest}
    case PiecesModel.Joined{left,right}: content_buffers(~Element,left,content_buffers(~Element,right,rest))

def content_line(~Element: Data,content: PiecesModel.Content<Element>,rest: Line<Element>) -> Line<Element>:
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}: Lined{buffer,first,rows,rest}
    case PiecesModel.Joined{left,right}: content_line(~Element,left,content_line(~Element,right,rest))

# Spans given the buffers in order; a span past the buffers gets the vacant
# buffer.
def spans_line(~Element: Data,+vacant: Buffer.Content<Element>,spans: Held.Spans,buffers: Buffers<Element>) -> Line<Element>:
  match spans buffers:
    case Held.Ended{} buffers: Unlined{}
    case Held.Spanned{first,count,rest} Unbuffered{}: Lined{vacant,first,count,spans_line(~Element,vacant,rest,Unbuffered{})}
    case Held.Spanned{first,count,rest} Buffered{buffer,more}: Lined{buffer,first,count,spans_line(~Element,vacant,rest,more)}

def LineAll(~Element: Data,~Context: Data,~Property: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,line: Line<Element>) -> Data:
  match line:
    case Unlined{}: Unit
    case Lined{buffer,first,rows,rest}: KernelGemm.Both<Property(context,first,rows,buffer),LineAll(~Element,~Context,~Property,context,rest)>

def all_lined(~Element: Data,~Context: Data,~Property: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,content: PiecesModel.Content<Element>,+rest: Line<Element>,+facts: PiecesModel.ContentAll(~Element,~Property(context),content),
  +after: LineAll(~Element,~Context,~Property,context,rest)) -> LineAll(~Element,~Context,~Property,context,content_line(~Element,content,rest)):
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}: KernelGemm.Both{facts,after}
    case PiecesModel.Joined{+left,+right}:
      KernelGemm.Both{+left_facts,+right_facts} = facts
      all_lined(~Element,~Context,~Property,context,left,content_line(~Element,right,rest),left_facts,all_lined(~Element,~Context,~Property,context,right,rest,right_facts,after))

def lined_rest(~Element: Data,~Context: Data,~Property: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,content: PiecesModel.Content<Element>,+rest: Line<Element>,+facts: LineAll(~Element,~Context,~Property,context,content_line(~Element,content,rest))) ->
  LineAll(~Element,~Context,~Property,context,rest):
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}:
      KernelGemm.Both{+fact,+after} = facts
      after
    case PiecesModel.Joined{+left,+right}:
      lined_rest(~Element,~Context,~Property,context,right,rest,lined_rest(~Element,~Context,~Property,context,left,content_line(~Element,right,rest),facts))

def lined_all(~Element: Data,~Context: Data,~Property: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,content: PiecesModel.Content<Element>,+rest: Line<Element>,+facts: LineAll(~Element,~Context,~Property,context,content_line(~Element,content,rest))) ->
  PiecesModel.ContentAll(~Element,~Property(context),content):
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}:
      KernelGemm.Both{+fact,+after} = facts
      fact
    case PiecesModel.Joined{+left,+right}:
      KernelGemm.Both{lined_all(~Element,~Context,~Property,context,left,content_line(~Element,right,rest),facts),
        lined_all(~Element,~Context,~Property,context,right,rest,lined_rest(~Element,~Context,~Property,context,left,content_line(~Element,right,rest),facts))}

# The line of contents is its spans with its buffers.
def lined_content(~Element: Data,+vacant: Buffer.Content<Element>,content: PiecesModel.Content<Element>,+spans: Held.Spans,+buffers: Buffers<Element>) ->
  {content_line(~Element,content,spans_line(~Element,vacant,spans,buffers)) ==
    spans_line(~Element,vacant,Held.layout_spans(PiecesModel.content_layout(~Element,content),spans),content_buffers(~Element,content,buffers)) : Line<Element>}:
  match content:
    case PiecesModel.PieceContent{buffer,first,rows}: {==}
    case PiecesModel.Joined{+left,+right}:
      +right_spans = Held.layout_spans(PiecesModel.content_layout(~Element,right),spans)
      +right_buffers = content_buffers(~Element,right,buffers)
      Equal.trans(Line<Element>,content_line(~Element,left,content_line(~Element,right,spans_line(~Element,vacant,spans,buffers))),
        content_line(~Element,left,spans_line(~Element,vacant,right_spans,right_buffers)),
        spans_line(~Element,vacant,Held.layout_spans(PiecesModel.content_layout(~Element,left),right_spans),content_buffers(~Element,left,right_buffers)),
        Equal.cong(Line<Element>,Line<Element>,line => content_line(~Element,left,line),content_line(~Element,right,spans_line(~Element,vacant,spans,buffers)),
          spans_line(~Element,vacant,right_spans,right_buffers),lined_content(~Element,vacant,right,spans,buffers)),
        lined_content(~Element,vacant,left,right_spans,right_buffers))

# The buffers left after ranges take theirs in order.
def regrouped_rest(~Element: Data,layout: Pieces.Ranges,buffers: Buffers<Element>) -> Buffers<Element>:
  match layout:
    case Pieces.Range{first,count}:
      match buffers:
        case Unbuffered{}: Unbuffered{}
        case Buffered{buffer,rest}: rest
    case Pieces.Split{left,right}: regrouped_rest(~Element,right,regrouped_rest(~Element,left,buffers))

# Contents of the ranges of a layout, each range taking the next buffer.
def regrouped_content(~Element: Data,+vacant: Buffer.Content<Element>,layout: Pieces.Ranges,+buffers: Buffers<Element>) -> PiecesModel.Content<Element>:
  match layout:
    case Pieces.Range{+first,+count}:
      match buffers:
        case Unbuffered{}: PiecesModel.PieceContent{vacant,first,count}
        case Buffered{buffer,rest}: PiecesModel.PieceContent{buffer,first,count}
    case Pieces.Split{+left,right}:
      PiecesModel.Joined{regrouped_content(~Element,vacant,left,buffers),regrouped_content(~Element,vacant,right,regrouped_rest(~Element,left,buffers))}

def regrouped_layout(~Element: Data,+vacant: Buffer.Content<Element>,layout: Pieces.Ranges,+buffers: Buffers<Element>) ->
  {PiecesModel.content_layout(~Element,regrouped_content(~Element,vacant,layout,buffers)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{+first,+count}:
      match buffers:
        case Unbuffered{}: {==}
        case Buffered{buffer,rest}: {==}
    case Pieces.Split{+left,+right}:
      +rest = regrouped_rest(~Element,left,buffers)
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,regrouped_content(~Element,vacant,left,buffers)),left,regrouped_layout(~Element,vacant,left,buffers)) :
        {Pieces.Split{_,PiecesModel.content_layout(~Element,regrouped_content(~Element,vacant,right,rest))} == Pieces.Split{left,right} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,regrouped_content(~Element,vacant,right,rest)),right,regrouped_layout(~Element,vacant,right,rest)) :
        {Pieces.Split{left,_} == Pieces.Split{left,right} : Pieces.Ranges}
      {==}

# The line of regrouped contents is the spans of the layout with the buffers.
def regrouped_line(~Element: Data,+vacant: Buffer.Content<Element>,layout: Pieces.Ranges,+spans: Held.Spans,+buffers: Buffers<Element>) ->
  {content_line(~Element,regrouped_content(~Element,vacant,layout,buffers),spans_line(~Element,vacant,spans,regrouped_rest(~Element,layout,buffers))) ==
    spans_line(~Element,vacant,Held.layout_spans(layout,spans),buffers) : Line<Element>}:
  match layout:
    case Pieces.Range{+first,+count}:
      match buffers:
        case Unbuffered{}: {==}
        case Buffered{buffer,rest}: {==}
    case Pieces.Split{+left,+right}:
      +middle = regrouped_rest(~Element,left,buffers)
      +left_content = regrouped_content(~Element,vacant,left,buffers)
      Equal.trans(Line<Element>,
        content_line(~Element,left_content,content_line(~Element,regrouped_content(~Element,vacant,right,middle),spans_line(~Element,vacant,spans,regrouped_rest(~Element,right,middle)))),
        content_line(~Element,left_content,spans_line(~Element,vacant,Held.layout_spans(right,spans),middle)),
        spans_line(~Element,vacant,Held.layout_spans(left,Held.layout_spans(right,spans)),buffers),
        Equal.cong(Line<Element>,Line<Element>,line => content_line(~Element,left_content,line),
          content_line(~Element,regrouped_content(~Element,vacant,right,middle),spans_line(~Element,vacant,spans,regrouped_rest(~Element,right,middle))),
          spans_line(~Element,vacant,Held.layout_spans(right,spans),middle),regrouped_line(~Element,vacant,right,spans,middle)),
        regrouped_line(~Element,vacant,left,Held.layout_spans(right,spans),buffers))

# What holds of every piece of contents holds of every piece of them regrouped
# by a layout of the same spans.
def regrouped_all(~Element: Data,~Context: Data,~Property: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +context: Context,+vacant: Buffer.Content<Element>,+content: PiecesModel.Content<Element>,+layout: Pieces.Ranges,
  +spans: {Held.layout_spans(PiecesModel.content_layout(~Element,content),Held.Ended{}) == Held.layout_spans(layout,Held.Ended{}) : Held.Spans},
  +facts: PiecesModel.ContentAll(~Element,~Property(context),content)) ->
  PiecesModel.ContentAll(~Element,~Property(context),regrouped_content(~Element,vacant,layout,content_buffers(~Element,content,Unbuffered{}))):
  +buffers = content_buffers(~Element,content,Unbuffered{})
  +regrouped = regrouped_content(~Element,vacant,layout,buffers)
  +lines = Equal.trans(Line<Element>,content_line(~Element,content,Unlined{}),
    spans_line(~Element,vacant,Held.layout_spans(PiecesModel.content_layout(~Element,content),Held.Ended{}),buffers),content_line(~Element,regrouped,Unlined{}),
    lined_content(~Element,vacant,content,Held.Ended{},Unbuffered{}),
    Equal.trans(Line<Element>,spans_line(~Element,vacant,Held.layout_spans(PiecesModel.content_layout(~Element,content),Held.Ended{}),buffers),
      spans_line(~Element,vacant,Held.layout_spans(layout,Held.Ended{}),buffers),content_line(~Element,regrouped,Unlined{}),
      Equal.cong(Held.Spans,Line<Element>,found => spans_line(~Element,vacant,found,buffers),Held.layout_spans(PiecesModel.content_layout(~Element,content),Held.Ended{}),
        Held.layout_spans(layout,Held.Ended{}),spans),
      Equal.sym(Line<Element>,content_line(~Element,regrouped,Unlined{}),spans_line(~Element,vacant,Held.layout_spans(layout,Held.Ended{}),buffers),
        regrouped_line(~Element,vacant,layout,Held.Ended{},buffers))))
  +lined = {%lines : LineAll(~Element,~Context,~Property,context,_)
    all_lined(~Element,~Context,~Property,context,content,Unlined{},facts,Unit{}) : LineAll(~Element,~Context,~Property,context,content_line(~Element,regrouped,Unlined{}))}
  lined_all(~Element,~Context,~Property,context,regrouped,Unlined{},lined)

# ---- The runtime regrouping ----

# Blocks in order, as the queue of the runtime holds them.
type Blocks<-Element: Data> is Data:
  Unblocked{}
  Blocked{buffer: Buffer.Model<Element>,rest: Blocks<Element>}

def blocks_queue(~Element: Data,blocks: Blocks<Element>) -> Pieces.Queue<Storage.Buffer<Element>>:
  match blocks:
    case Unblocked{}: Pieces.Empty{}
    case Blocked{buffer,rest}: Pieces.Queued{Buffer.pack(~Element,buffer),blocks_queue(~Element,rest)}

def blocks_buffers(~Element: Data,blocks: Blocks<Element>) -> Buffers<Element>:
  match blocks:
    case Unblocked{}: Unbuffered{}
    case Blocked{buffer,rest}: Buffered{Buffer.contents(~Element,Buffer.pack(~Element,buffer)),blocks_buffers(~Element,rest)}

def model_blocks(~Element: Data,model: PiecesModel.Model<Element>,rest: Blocks<Element>) -> Blocks<Element>:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: Blocked{buffer,rest}
    case PiecesModel.Fork{left,right}: model_blocks(~Element,left,model_blocks(~Element,right,rest))

# The blocks left after ranges take theirs in order.
def blocks_after(~Element: Data,layout: Pieces.Ranges,blocks: Blocks<Element>) -> Blocks<Element>:
  match layout:
    case Pieces.Range{first,count}:
      match blocks:
        case Unblocked{}: Unblocked{}
        case Blocked{buffer,rest}: rest
    case Pieces.Split{left,right}: blocks_after(~Element,right,blocks_after(~Element,left,blocks))

def after_buffers(~Element: Data,layout: Pieces.Ranges,+blocks: Blocks<Element>) ->
  {blocks_buffers(~Element,blocks_after(~Element,layout,blocks)) == regrouped_rest(~Element,layout,blocks_buffers(~Element,blocks)) : Buffers<Element>}:
  match layout:
    case Pieces.Range{first,count}:
      match blocks:
        case Unblocked{}: {==}
        case Blocked{buffer,rest}: {==}
    case Pieces.Split{+left,+right}:
      Equal.trans(Buffers<Element>,blocks_buffers(~Element,blocks_after(~Element,right,blocks_after(~Element,left,blocks))),
        regrouped_rest(~Element,right,blocks_buffers(~Element,blocks_after(~Element,left,blocks))),
        regrouped_rest(~Element,right,regrouped_rest(~Element,left,blocks_buffers(~Element,blocks))),
        after_buffers(~Element,right,blocks_after(~Element,left,blocks)),
        Equal.cong(Buffers<Element>,Buffers<Element>,buffers => regrouped_rest(~Element,right,buffers),blocks_buffers(~Element,blocks_after(~Element,left,blocks)),
          regrouped_rest(~Element,left,blocks_buffers(~Element,blocks)),after_buffers(~Element,left,blocks)))

def queued_blocks(~Element: Data,+model: PiecesModel.Model<Element>,+blocks: Blocks<Element>) ->
  {Pieces.queued(~Storage.Buffer<Element>,PiecesModel.pack(~Element,model),blocks_queue(~Element,blocks)) == blocks_queue(~Element,model_blocks(~Element,model,blocks)) : Pieces.Queue<Storage.Buffer<Element>>}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      %Equal.sym(Pieces.Queue<Storage.Buffer<Element>>,Pieces.queued(~Storage.Buffer<Element>,PiecesModel.pack(~Element,right),blocks_queue(~Element,blocks)),
          blocks_queue(~Element,model_blocks(~Element,right,blocks)),queued_blocks(~Element,right,blocks)) :
        {Pieces.queued(~Storage.Buffer<Element>,PiecesModel.pack(~Element,left),_) == blocks_queue(~Element,model_blocks(~Element,left,model_blocks(~Element,right,blocks))) : Pieces.Queue<Storage.Buffer<Element>>}
      queued_blocks(~Element,left,model_blocks(~Element,right,blocks))

def model_buffers(~Element: Data,+model: PiecesModel.Model<Element>,+blocks: Blocks<Element>) ->
  {content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),blocks_buffers(~Element,blocks)) ==
    blocks_buffers(~Element,model_blocks(~Element,model,blocks)) : Buffers<Element>}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      %Equal.sym(Buffers<Element>,content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,right)),blocks_buffers(~Element,blocks)),
          blocks_buffers(~Element,model_blocks(~Element,right,blocks)),model_buffers(~Element,right,blocks)) :
        {content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,left)),_) ==
          blocks_buffers(~Element,model_blocks(~Element,left,model_blocks(~Element,right,blocks))) : Buffers<Element>}
      model_buffers(~Element,left,model_blocks(~Element,right,blocks))

def built(~Element: Data,+fill: Element,layout: Pieces.Ranges,queue: Pieces.Queue<Storage.Buffer<Element>>) -> Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>:
  Pieces.build(~Storage.Buffer<Element>,~Pieces.Queue<Storage.Buffer<Element>>,~Element,~Pieces.dequeued(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element)),layout,fill,queue)

def fork_built(~Element: Data,left: Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair: Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>) ->
  {(Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair),Partition.Fork{left,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair)}) ==
    Storage.bind_pair(~Pieces.Queue<Storage.Buffer<Element>>,~Partition.Tree<&1,Storage.Buffer<Element>,Unit>,~Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair,state => right => (state,Partition.Fork{left,right})) : Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>}:
  (state,right) = pair
  {==}

def split_built(~Element: Data,+fill: Element,+right: Pieces.Ranges,pair: Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>) ->
  {(Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair))),
      Partition.Fork{Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair),Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair)))}) ==
    Storage.bind_pair(~Pieces.Queue<Storage.Buffer<Element>>,~Partition.Tree<&1,Storage.Buffer<Element>,Unit>,~Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>,pair,state => left =>
      Storage.bind_pair(~Pieces.Queue<Storage.Buffer<Element>>,~Partition.Tree<&1,Storage.Buffer<Element>,Unit>,~Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,state),state => right => (state,Partition.Fork{left,right}))) : Pieces.Queue<Storage.Buffer<Element>> & Partition.Tree<&1,Storage.Buffer<Element>,Unit>}:
  (state,left) = pair
  fork_built(~Element,left,built(~Element,fill,right,state))

def built_rest(~Element: Data,+fill: Element,layout: Pieces.Ranges,+blocks: Blocks<Element>) ->
  {Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,layout,blocks_queue(~Element,blocks))) == blocks_queue(~Element,blocks_after(~Element,layout,blocks)) : Pieces.Queue<Storage.Buffer<Element>>}:
  match layout:
    case Pieces.Range{+first,+count}:
      match blocks:
        case Unblocked{}: {==}
        case Blocked{buffer,rest}: {==}
    case Pieces.Split{+left,+right}:
      %split_built(~Element,fill,right,built(~Element,fill,left,blocks_queue(~Element,blocks))) :
        {Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,_) == blocks_queue(~Element,blocks_after(~Element,right,blocks_after(~Element,left,blocks))) : Pieces.Queue<Storage.Buffer<Element>>}
      %Equal.sym(Pieces.Queue<Storage.Buffer<Element>>,Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,left,blocks_queue(~Element,blocks))),blocks_queue(~Element,blocks_after(~Element,left,blocks)),
          built_rest(~Element,fill,left,blocks)) :
        {Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,_)) == blocks_queue(~Element,blocks_after(~Element,right,blocks_after(~Element,left,blocks))) : Pieces.Queue<Storage.Buffer<Element>>}
      built_rest(~Element,fill,right,blocks_after(~Element,left,blocks))

def built_content(~Element: Data,+fill: Element,layout: Pieces.Ranges,+blocks: Blocks<Element>) ->
  {PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,layout,blocks_queue(~Element,blocks)))) ==
    regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),layout,blocks_buffers(~Element,blocks)) : PiecesModel.Content<Element>}:
  match layout:
    case Pieces.Range{+first,+count}:
      match blocks:
        case Unblocked{}: {==}
        case Blocked{buffer,rest}: {==}
    case Pieces.Split{+left,+right}:
      +after = blocks_after(~Element,left,blocks)
      +left_content = regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),left,blocks_buffers(~Element,blocks))
      +right_content = regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),right,regrouped_rest(~Element,left,blocks_buffers(~Element,blocks)))
      %split_built(~Element,fill,right,built(~Element,fill,left,blocks_queue(~Element,blocks))) :
        {PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,_)) == PiecesModel.Joined{left_content,right_content} : PiecesModel.Content<Element>}
      %Equal.sym(Pieces.Queue<Storage.Buffer<Element>>,Pair.fst(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,left,blocks_queue(~Element,blocks))),blocks_queue(~Element,after),built_rest(~Element,fill,left,blocks)) :
        {PiecesModel.Joined{PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,left,blocks_queue(~Element,blocks)))),
          PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,_)))} == PiecesModel.Joined{left_content,right_content} : PiecesModel.Content<Element>}
      %Equal.sym(PiecesModel.Content<Element>,PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,left,blocks_queue(~Element,blocks)))),left_content,
          built_content(~Element,fill,left,blocks)) :
        {PiecesModel.Joined{_,PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,blocks_queue(~Element,after))))} ==
          PiecesModel.Joined{left_content,right_content} : PiecesModel.Content<Element>}
      %Equal.sym(PiecesModel.Content<Element>,PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,blocks_queue(~Element,after)))),right_content,
          Equal.trans(PiecesModel.Content<Element>,PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,right,blocks_queue(~Element,after)))),
            regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),right,blocks_buffers(~Element,after)),right_content,
            built_content(~Element,fill,right,after),
            Equal.cong(Buffers<Element>,PiecesModel.Content<Element>,buffers => regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),right,buffers),blocks_buffers(~Element,after),
              regrouped_rest(~Element,left,blocks_buffers(~Element,blocks)),after_buffers(~Element,left,blocks)))) :
        {PiecesModel.Joined{left_content,_} == PiecesModel.Joined{left_content,right_content} : PiecesModel.Content<Element>}
      {==}

# The runtime regrouping gives every range the next block of the tree.
def regrouped_contents(~Element: Data,+fill: Element,+layout: Pieces.Ranges,+model: PiecesModel.Model<Element>) ->
  {PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),layout,fill,PiecesModel.pack(~Element,model))) ==
    regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),layout,content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),Unbuffered{})) : PiecesModel.Content<Element>}:
  +blocks = model_blocks(~Element,model,Unblocked{})
  %Equal.sym(Pieces.Queue<Storage.Buffer<Element>>,Pieces.queued(~Storage.Buffer<Element>,PiecesModel.pack(~Element,model),Pieces.Empty{}),blocks_queue(~Element,blocks),queued_blocks(~Element,model,Unblocked{})) :
    {PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,layout,_))) ==
      regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),layout,content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),Unbuffered{})) : PiecesModel.Content<Element>}
  %Equal.sym(Buffers<Element>,content_buffers(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),Unbuffered{}),blocks_buffers(~Element,blocks),
      model_buffers(~Element,model,Unblocked{})) :
    {PiecesModel.contents(~Element,Pair.snd(Pieces.Queue<Storage.Buffer<Element>>,Partition.Tree<&1,Storage.Buffer<Element>,Unit>,built(~Element,fill,layout,blocks_queue(~Element,blocks)))) ==
      regrouped_content(~Element,Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill)),layout,_) : PiecesModel.Content<Element>}
  built_content(~Element,fill,layout,blocks)

def ready_at(~Element: Data,+context: Unit,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  PiecesModel.ready_piece(~Element,first,rows,buffer)

def ready_into(~Element: Data,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,fact: PiecesModel.ready_piece(~Element,first,rows,buffer)) ->
  ready_at(~Element,Unit{},first,rows,buffer):
  fact

def ready_from(~Element: Data,+context: Unit,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,fact: ready_at(~Element,context,first,rows,buffer)) ->
  PiecesModel.ready_piece(~Element,first,rows,buffer):
  fact

# Pieces of a grid cut at the same places as the target's, regrouped by the
# ranges of the target, are the pieces of the target.
def regrouped_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+model: PiecesModel.Model<Element>,
  +source: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers}),
  +count: U32,+source_counted: {Tensors.count_of(source) == Some{count} : Maybe<&2,U32>},+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonzero: {U32.is_zero(count) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +same: {Grid.same_pieces(Tensors.grid_of(source),Tensors.grid_of(target)) == True{} : Bool}) ->
  Global.Contract(~Element,~Logical,~original,logical,
    Tensors.Tensor{Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),Pieces.ranges_of_grid(Tensors.grid_of(target)),fill,
      PiecesModel.pack(~Element,model)),target,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{+normal,+values} = contract
  KernelGemm.Both{+formed,+rest} = normal
  Content.WellFormed{+fields,+ready,+coverage} = formed
  KernelGemm.Both{+counted,+canonical} = rest
  +ranges = Pieces.ranges_of_grid(Tensors.grid_of(target))
  +vacant = Buffer.contents(~Element,Pieces.vacant_buffer(~Element,fill))
  +content = PiecesModel.contents(~Element,PiecesModel.pack(~Element,model))
  +regrouped = regrouped_content(~Element,vacant,ranges,content_buffers(~Element,content,Unbuffered{}))
  +total = ShapeLaws.elements(target)
  +elements = Equal.trans(Nat,ShapeLaws.elements(source),U32.to_nat(count),total,Global.whole_counted(source,count,source_counted),
    Equal.sym(Nat,total,U32.to_nat(count),Global.whole_counted(target,count,target_counted)))
  +source_nonempty = {%Equal.sym(Nat,ShapeLaws.elements(source),U32.to_nat(count),Global.whole_counted(source,count,source_counted)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Arithmetic.word_nonzero_positive(count,nonzero) : {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}}
  +target_nonempty = {%Equal.sym(Nat,total,U32.to_nat(count),Global.whole_counted(target,count,target_counted)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Arithmetic.word_nonzero_positive(count,nonzero) : {Nat.is_lt(0n,total) == True{} : Bool}}
  +spans = Equal.trans(Held.Spans,Held.layout_spans(PiecesModel.content_layout(~Element,content),Held.Ended{}),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(source)),Held.Ended{}),Held.layout_spans(ranges,Held.Ended{}),
    Equal.cong(Pieces.Ranges,Held.Spans,layout => Held.layout_spans(layout,Held.Ended{}),PiecesModel.content_layout(~Element,content),
      Pieces.ranges_of_grid(Tensors.grid_of(source)),canonical),
    same_cuts(limits,source,target,StorageCount.counted_room(limits,source,Global.Counted{count,source_counted},default_capacity),source_nonempty,
      StorageCount.counted_room(limits,target,Global.Counted{count,target_counted},default_capacity),target_nonempty,same))
  +contents = Equal.sym(PiecesModel.Content<Element>,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model))),regrouped,regrouped_contents(~Element,fill,ranges,model))
  +moved_values = {%contents : Content.Values(~Element,~Logical,~original,logical,_)
    regrouped_all(~Element,~Logical,~Content.piece_values(~Element,~Logical,~original),logical,vacant,content,ranges,spans,values) :
    Content.Values(~Element,~Logical,~original,logical,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model))))}
  +moved_fields = {%contents : Content.Contents(~Element,total,_)
    regrouped_all(~Element,~Nat,~Content.piece_fields(~Element),total,vacant,content,ranges,spans,
      {%elements : Content.Contents(~Element,_,content) fields : Content.Contents(~Element,total,content)}) :
    Content.Contents(~Element,total,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model))))}
  +moved_ready = {%contents : PiecesModel.ContentReady(~Element,_)
    PiecesModel.content_plain(~Element,~Unit,~ready_at(~Element),~PiecesModel.ready_piece(~Element),~ready_from(~Element),Unit{},regrouped,
      regrouped_all(~Element,~Unit,~ready_at(~Element),Unit{},vacant,content,ranges,spans,
        PiecesModel.content_map(~Element,~PiecesModel.ready_piece(~Element),~ready_at(~Element,Unit{}),~ready_into(~Element),content,ready))) : PiecesModel.ContentReady(~Element,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model))))}
  +laid = Equal.trans(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model)))),PiecesModel.content_layout(~Element,regrouped),ranges,
    Equal.cong(PiecesModel.Content<Element>,Pieces.Ranges,found => PiecesModel.content_layout(~Element,found),PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model))),regrouped,
      regrouped_contents(~Element,fill,ranges,model)),
    regrouped_layout(~Element,vacant,ranges,content_buffers(~Element,content,Unbuffered{})))
  +moved_coverage = {%Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model)))),ranges,laid) : PiecesPlan.Coverage(_,0n,total)
    layout_coverage(target,target_layout(limits,target,count,target_counted,nonzero,default_capacity)) : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.contents(~Element,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model)))),0n,total)}
  Global.contract(~Element,~Logical,~original,logical,target,fill,workers,Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),ranges,fill,PiecesModel.pack(~Element,model)),Global.Counted{count,target_counted},laid,
    KernelGemm.Both{moved_values,Content.WellFormed{moved_fields,moved_ready,moved_coverage}})

# The runtime keeps the PiecesModel.pack(~Element,model) of a grid of the same ranges, regroups the
# PiecesModel.pack(~Element,model) of a grid cut at the same places and copies the cells into the
# PiecesModel.pack(~Element,model) of another.
def relabel_when_shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,same: Bool,regrouped: Bool,+logical: Logical,+limits: Machine.Limits,
  +model: PiecesModel.Model<Element>,+source: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +classified: {Grid.same_grid(Tensors.grid_of(source),Tensors.grid_of(target)) == same : Bool},
  +grouping: {Grid.same_pieces(Tensors.grid_of(source),Tensors.grid_of(target)) == regrouped : Bool},
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers}),
  +count: U32,+source_counted: {Tensors.count_of(source) == Some{count} : Maybe<&2,U32>},+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonzero: {U32.is_zero(count) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Logical,~original,logical,target,Tensors.relabel_when(~Element,same,regrouped,PiecesModel.pack(~Element,model),target,fill,workers)):
  match same regrouped:
    case True{} regrouped:
      Global.shaped(~Element,~Logical,~original,logical,target,Tensors.Tensor{PiecesModel.pack(~Element,model),target,fill,Tensors.Usable{},workers},{==},{==},
        kept_contract(~Element,~Logical,~original,logical,PiecesModel.pack(~Element,model),source,target,fill,workers,contract,count,source_counted,target_counted,classified))
    case False{} True{}:
      Global.shaped(~Element,~Logical,~original,logical,target,
        Tensors.Tensor{Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),Pieces.ranges_of_grid(Tensors.grid_of(target)),fill,
          PiecesModel.pack(~Element,model)),target,fill,Tensors.Usable{},workers},{==},{==},
        regrouped_contract(~Element,~Logical,~original,logical,limits,model,source,target,fill,workers,contract,count,source_counted,target_counted,nonzero,default_capacity,grouping))
    case False{} False{}:
      Global.shaped(~Element,~Logical,~original,logical,target,
        Tensors.Tensor{Tensors.flat_moved(~Element,Tensors.grid_of(target),fill,PiecesModel.pack(~Element,model)),target,fill,Tensors.Usable{},workers},{==},{==},
        moved_contract(~Element,~Logical,~original,logical,limits,model,source,target,fill,workers,contract,count,source_counted,target_counted,nonzero,default_capacity))

# The cells of a ready tensor of nonzero cells under another shape of as many
# cells hold the values they held.
def relabeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,
  -tensor: Tensors.Tensor<Element>,+shaped: Global.Shaped(~Element,~Logical,~original,logical,source,tensor),
  +count: U32,+source_counted: {Tensors.count_of(source) == Some{count} : Maybe<&2,U32>},+target_counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},
  +nonzero: {U32.is_zero(count) == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Logical,~original,logical,target,Tensors.relabeled(~Element,tensor,target)):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},+contract} = shaped
  %equation : Global.Shaped(~Element,~Logical,~original,logical,target,Tensors.relabeled(~Element,_,target))
  relabel_when_shaped(~Element,~Logical,~original,Grid.same_grid(Tensors.grid_of(source),Tensors.grid_of(target)),Grid.same_pieces(Tensors.grid_of(source),Tensors.grid_of(target)),
    logical,limits,model,source,target,fill,workers,{==},{==},
    {%Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers},tensor,equation) :
        Global.Contract(~Element,~Logical,~original,logical,_)
      contract : Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers})},
    count,source_counted,target_counted,nonzero,default_capacity)

# ---- Reshape ----

def count_bounded(+expected: Nat,+count: U32,+evidence: Count.Count<expected,count>) -> {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{+exact,+bounded} = evidence
  bounded

# A shape of as many elements as a counted shape counts as many.
def counted_same(+shape: List<&2,U32>,+other: List<&2,U32>,+counted: Global.Counted<shape>,+elements: {ShapeLaws.elements(shape) == ShapeLaws.elements(other) : Nat}) ->
  {Tensors.count_of(other) == Some{Global.count(shape,counted)} : Maybe<&2,U32>}:
  +count = Global.count(shape,counted)
  CountCompletion.count_of(other,count,
    Equal.trans(Nat,ShapeLaws.elements(other),ShapeLaws.elements(shape),U32.to_nat(count),Equal.sym(Nat,ShapeLaws.elements(shape),ShapeLaws.elements(other),elements),
      Global.whole_counted(shape,count,Global.accepted(shape,counted))),
    count_bounded(ShapeLaws.elements(shape),count,Global.evidence(shape,counted)))

# The cells of a ready tensor of nonzero cells reshaped to extents of as many
# cells hold the values they held.
def reshaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,
  -tensor: Tensors.Tensor<Element>,+shaped: Global.Shaped(~Element,~Logical,~original,logical,source,tensor),
  +elements: {ShapeLaws.elements(source) == ShapeLaws.elements(target) : Nat},+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Logical,~original,logical,target,Layout.reshape(~Element,tensor,target)):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},+contract} = shaped
  %equation : Global.Shaped(~Element,~Logical,~original,logical,target,Layout.reshape(~Element,_,target))
  +held = {%Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers},tensor,equation) :
      Global.Contract(~Element,~Logical,~original,logical,_)
    contract : Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers})}
  +counted = Global.contract_counted(~Element,~Logical,~original,logical,PiecesModel.pack(~Element,model),source,fill,workers,held)
  +count = Global.count(source,counted)
  +source_counted = Global.accepted(source,counted)
  +target_counted = counted_same(source,target,counted,elements)
  +nonzero = Arithmetic.word_positive_nonzero(count,
    {%Global.whole_counted(source,count,source_counted) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}})
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(source),Some{count},source_counted) :
    Global.Shaped(~Element,~Logical,~original,logical,target,
      Layout.reshape_status(~Element,Tensors.Usable{},Layout.same_count(_,Tensors.count_of(target)),PiecesModel.pack(~Element,model),source,target,fill,workers))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(target),Some{count},target_counted) :
    Global.Shaped(~Element,~Logical,~original,logical,target,
      Layout.reshape_status(~Element,Tensors.Usable{},Layout.same_count(Some{count},_),PiecesModel.pack(~Element,model),source,target,fill,workers))
  %Equal.sym(Bool,U32.is_eq(count,count),True{},Arithmetic.equal_reflexive(count)) :
    Global.Shaped(~Element,~Logical,~original,logical,target,Layout.reshape_status(~Element,Tensors.Usable{},_,PiecesModel.pack(~Element,model),source,target,fill,workers))
  relabeled(~Element,~Logical,~original,logical,limits,source,target,Tensors.Tensor{PiecesModel.pack(~Element,model),source,fill,Tensors.Usable{},workers},
    KernelGemm.Both{Global.Usable{model,fill,workers,{==}},held},count,source_counted,target_counted,nonzero,default_capacity)
