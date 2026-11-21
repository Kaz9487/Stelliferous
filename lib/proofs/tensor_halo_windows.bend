# The windows of a window operation: every task holds the cells of its reach
# with the values of the input tensor, a halo copied from the input pieces and
# its own pieces after it.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../tensor_move.bend as Move
import ../tensor_halo.bend as Halo
import ../traversal_partition.bend as Partition
import ../storage_allocation.bend as Allocation
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_content.bend as Content
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./encoded_coordinates.bend as Coordinates
import ./tensor_grid_pieces.bend as Laws
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ../tensor.bend as Tensors
import ../kernel_gemm.bend as KernelGemm
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws

# ---- What the walk needs of the grid and the reach ----

def span_start(+span: Tasks.Span) -> U32:
  Tasks.Span{start,length} = span
  start

def span_of(+start: U32,+length: U32) -> Tasks.Span:
  Tasks.Span{start,length}

def span_end(+span: Tasks.Span) -> U32:
  Tasks.Span{+start,+length} = span
  (start + length : U32)

# Every reach of a task lies inside the tensor and ends at or after it starts.
def Reaches(~Task: Data,~reach: Halo.Reach(~Task),+task: Task,tasks: Pieces.Ranges,+total: Nat) -> Data:
  match tasks:
    case Pieces.Range{+first,+count}: Held.Both<{Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,first,count))),Nat.sub(U32.to_nat(span_end(reach(task,first,count))),U32.to_nat(span_start(reach(task,first,count))))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(span_start(reach(task,first,count))),U32.to_nat(span_end(reach(task,first,count)))) == True{} : Bool}>
    case Pieces.Split{+left,+right}: Held.Both<Reaches(~Task,~reach,task,left,total),Reaches(~Task,~reach,task,right,total)>

# A span from low to high: its cells end at high.
def span_fits(+limits: Machine.Limits,+low: U32,+high: U32,+total: Nat,+ordered: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(high),total) == True{} : Bool},+room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Held.Both<{Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(U32.to_nat((low + (high - low : U32) : U32)),U32.to_nat(low))),total) == True{} : Bool},
    {Nat.is_le(U32.to_nat(low),U32.to_nat((low + (high - low : U32) : U32))) == True{} : Bool}>:
  +fits = Limits.capacity_fits(limits)
  +high_room = Order.transitive(U32.to_nat(high),total,U32.to_nat(Machine.capacity(limits)),inside,room)
  +difference = Arithmetic.subtract(high,low,Machine.capacity(limits),fits,ordered,Order.transitive(Nat.sub(U32.to_nat(high),U32.to_nat(low)),U32.to_nat(high),U32.to_nat(Machine.capacity(limits)),Order.sub_le(U32.to_nat(high),U32.to_nat(low)),high_room))
  +restored = Order.restore_shift(U32.to_nat(low),U32.to_nat(high),ordered)
  +moved = Equal.trans(Nat,Nat.add(U32.to_nat(low),U32.to_nat((high - low : U32))),Nat.add(U32.to_nat(low),Nat.sub(U32.to_nat(high),U32.to_nat(low))),U32.to_nat(high),
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(low),value),U32.to_nat((high - low : U32)),Nat.sub(U32.to_nat(high),U32.to_nat(low)),difference),restored)
  +sum = Arithmetic.add(low,(high - low : U32),Machine.capacity(limits),fits,Held.nat_le_moved(Nat.add(U32.to_nat(low),U32.to_nat((high - low : U32))),U32.to_nat(high),U32.to_nat(Machine.capacity(limits)),moved,high_room))
  +ended = Equal.trans(Nat,U32.to_nat((low + (high - low : U32) : U32)),Nat.add(U32.to_nat(low),U32.to_nat((high - low : U32))),U32.to_nat(high),sum,moved)
  %Equal.sym(Nat,U32.to_nat((low + (high - low : U32) : U32)),U32.to_nat(high),ended) :
    Held.Both<{Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(_,U32.to_nat(low))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(low),_) == True{} : Bool}>
  Held.Both{Held.nat_le_moved(Nat.add(U32.to_nat(low),Nat.sub(U32.to_nat(high),U32.to_nat(low))),U32.to_nat(high),total,restored,inside),ordered}

# Where the first count ranges of a layout from low end.
def span_boundary(count: Nat,spans: Held.Spans,+low: Nat) -> Nat:
  match count spans:
    case 1n+rest Held.Spanned{first,+pieces,below}: span_boundary(rest,below,Nat.add(low,U32.to_nat(pieces)))
    case count spans: low

# The layout of the tensor is the pieces of the grid: the range k starts where
# piece k starts, and the ranges up to the piece from a cell end at or after it.
def LayoutStarts(+grid: Grid.Grid,+layout: Held.Spans) -> Type:
  @+piece: U32 -> @+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool} ->
    {span_boundary(U32.to_nat(piece),layout,0n) == U32.to_nat(Grid.start(grid,piece)) : Nat}

def ReachedAfter(+grid: Grid.Grid,+layout: Held.Spans,+total: Nat) -> Type:
  @+index: U32 -> @+inside: {Nat.is_le(U32.to_nat(index),total) == True{} : Bool} ->
    {Nat.is_le(U32.to_nat(index),span_boundary(U32.to_nat(Halo.piece_from(grid,index)),layout,0n)) == True{} : Bool}

# The layout from a piece on, low cells before it: every range starts where its
# piece starts, and the layout ends after the last piece at the cells of the grid.
def Along(+grid: Grid.Grid,spans: Held.Spans,+piece: Nat,+low: Nat) -> Data:
  match spans:
    case Held.Ended{}: Held.Both<{piece == U32.to_nat(Grid.pieces_of(grid)) : Nat},{low == Laws.grid_cells(grid) : Nat}>
    case Held.Spanned{first,+count,rest}:
      Held.Both<{Nat.is_lt(piece,U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool},
        Held.Both<{low == U32.to_nat(Grid.start(grid,U32.from_nat(piece))) : Nat},Along(grid,rest,1n+piece,Nat.add(low,U32.to_nat(count)))>>

def along_last(+grid: Grid.Grid,+piece: Nat,+low: Nat,+along: Along(grid,Held.Ended{},piece,low)) -> {piece == U32.to_nat(Grid.pieces_of(grid)) : Nat}:
  Held.Both{last,cells} = along
  last

def along_cells(+grid: Grid.Grid,+piece: Nat,+low: Nat,+along: Along(grid,Held.Ended{},piece,low)) -> {low == Laws.grid_cells(grid) : Nat}:
  Held.Both{last,cells} = along
  cells

def along_inside(+grid: Grid.Grid,+first: U32,+size: U32,+rest: Held.Spans,+piece: Nat,+low: Nat,+along: Along(grid,Held.Spanned{first,size,rest},piece,low)) -> {Nat.is_lt(piece,U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}:
  Held.Both{inside,more} = along
  inside

def along_more(+grid: Grid.Grid,+first: U32,+size: U32,+rest: Held.Spans,+piece: Nat,+low: Nat,+along: Along(grid,Held.Spanned{first,size,rest},piece,low)) -> Held.Both<{low == U32.to_nat(Grid.start(grid,U32.from_nat(piece))) : Nat},Along(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size)))>:
  Held.Both{inside,more} = along
  more

def along_here(+grid: Grid.Grid,+first: U32,+size: U32,+rest: Held.Spans,+piece: Nat,+low: Nat,+more: Held.Both<{low == U32.to_nat(Grid.start(grid,U32.from_nat(piece))) : Nat},Along(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size)))>) -> {low == U32.to_nat(Grid.start(grid,U32.from_nat(piece))) : Nat}:
  Held.Both{here,later} = more
  here

def along_later(+grid: Grid.Grid,+first: U32,+size: U32,+rest: Held.Spans,+piece: Nat,+low: Nat,+more: Held.Both<{low == U32.to_nat(Grid.start(grid,U32.from_nat(piece))) : Nat},Along(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size)))>) -> Along(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size))):
  Held.Both{here,later} = more
  later

def along_starts(+grid: Grid.Grid,+spans: Held.Spans,+piece: Nat,+low: Nat,+along: Along(grid,spans,piece,low),+count: Nat,
  +inside: {Nat.is_lt(Nat.add(piece,count),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}) ->
  {span_boundary(count,spans,low) == U32.to_nat(Grid.start(grid,U32.from_nat(Nat.add(piece,count)))) : Nat}:
  match spans count:
    case Held.Ended{} count:
      Empty.absurd({span_boundary(count,Held.Ended{},low) == U32.to_nat(Grid.start(grid,U32.from_nat(Nat.add(piece,count)))) : Nat},
        Held.lt_le_absurd(Nat.add(piece,count),U32.to_nat(Grid.pieces_of(grid)),inside,
          Held.nat_le_moved(U32.to_nat(Grid.pieces_of(grid)),piece,Nat.add(piece,count),Equal.sym(Nat,piece,U32.to_nat(Grid.pieces_of(grid)),along_last(grid,piece,low,along)),
            Order.le_add(piece,count))))
    case Held.Spanned{+first,+size,+rest} 0n:
      %Equal.sym(Nat,Nat.add(piece,0n),piece,Addition.zero_right(piece)) : {low == U32.to_nat(Grid.start(grid,U32.from_nat(_))) : Nat}
      along_here(grid,first,size,rest,piece,low,along_more(grid,first,size,rest,piece,low,along))
    case Held.Spanned{+first,+size,+rest} 1n+ +more:
      %Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)) : {span_boundary(more,rest,Nat.add(low,U32.to_nat(size))) == U32.to_nat(Grid.start(grid,U32.from_nat(_))) : Nat}
      along_starts(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size)),along_later(grid,first,size,rest,piece,low,along_more(grid,first,size,rest,piece,low,along)),more,
        {%Addition.successor_right(piece,more) : {Nat.is_lt(_,U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool} inside :
          {Nat.is_lt(1n+Nat.add(piece,more),U32.to_nat(Grid.pieces_of(grid))) == True{} : Bool}})

def along_end(+grid: Grid.Grid,+spans: Held.Spans,+piece: Nat,+low: Nat,+along: Along(grid,spans,piece,low),+count: Nat,
  +exact: {Nat.add(piece,count) == U32.to_nat(Grid.pieces_of(grid)) : Nat}) -> {span_boundary(count,spans,low) == Laws.grid_cells(grid) : Nat}:
  match spans count:
    case Held.Ended{} 0n: along_cells(grid,piece,low,along)
    case Held.Ended{} 1n+ +more: along_cells(grid,piece,low,along)
    case Held.Spanned{+first,+size,+rest} 0n:
      Empty.absurd({low == Laws.grid_cells(grid) : Nat},
        Held.lt_le_absurd(piece,U32.to_nat(Grid.pieces_of(grid)),along_inside(grid,first,size,rest,piece,low,along),
          Held.nat_le_moved(U32.to_nat(Grid.pieces_of(grid)),Nat.add(piece,0n),piece,Equal.sym(Nat,Nat.add(piece,0n),U32.to_nat(Grid.pieces_of(grid)),exact),
            Held.nat_le_moved(Nat.add(piece,0n),piece,piece,Addition.zero_right(piece),Order.reflexive(piece)))))
    case Held.Spanned{+first,+size,+rest} 1n+ +more:
      along_end(grid,rest,1n+piece,Nat.add(low,U32.to_nat(size)),along_later(grid,first,size,rest,piece,low,along_more(grid,first,size,rest,piece,low,along)),
        more,Equal.trans(Nat,1n+Nat.add(piece,more),Nat.add(piece,1n+more),U32.to_nat(Grid.pieces_of(grid)),
          Equal.sym(Nat,Nat.add(piece,1n+more),1n+Nat.add(piece,more),Addition.successor_right(piece,more)),exact))

# The grid of the tensor, its cells and its layout.
type Laid<-limits: Machine.Limits,-grid: Grid.Grid,-layout: Held.Spans,-total: Nat> is Data:
  Laid{facts: Laws.GridFacts<limits,grid>,cells: {total == Laws.grid_cells(grid) : Nat},along: Along(grid,layout,0n,0n)}

def laid_facts(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> Laws.GridFacts<limits,grid>:
  Laid{facts,cells,along} = laid
  facts

def laid_cells(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> {total == Laws.grid_cells(grid) : Nat}:
  Laid{facts,cells,along} = laid
  cells

def laid_along(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> Along(grid,layout,0n,0n):
  Laid{facts,cells,along} = laid
  along

def after_of(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> Laws.StartsAfter(grid,total):
  %Equal.sym(Nat,total,Laws.grid_cells(grid),laid_cells(limits,grid,layout,total,laid)) : Laws.StartsAfter(grid,_)
  Laws.grid_after(limits,grid,laid_facts(limits,grid,layout,total,laid))

def before_of(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> Laws.StartsBefore(grid,total):
  %Equal.sym(Nat,total,Laws.grid_cells(grid),laid_cells(limits,grid,layout,total,laid)) : Laws.StartsBefore(grid,_)
  Laws.grid_before(limits,grid,laid_facts(limits,grid,layout,total,laid))

def monotone_of(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> Laws.FromMonotone(grid,total):
  %Equal.sym(Nat,total,Laws.grid_cells(grid),laid_cells(limits,grid,layout,total,laid)) : Laws.FromMonotone(grid,_)
  Laws.grid_monotone(limits,grid,laid_facts(limits,grid,layout,total,laid))

def starts_of(+grid: Grid.Grid,+layout: Held.Spans,+along: Along(grid,layout,0n,0n)) -> LayoutStarts(grid,layout):
  +piece => +inside => Equal.trans(Nat,span_boundary(U32.to_nat(piece),layout,0n),U32.to_nat(Grid.start(grid,U32.from_nat(U32.to_nat(piece)))),U32.to_nat(Grid.start(grid,piece)),
    along_starts(grid,layout,0n,0n,along,U32.to_nat(piece),inside),
    Equal.cong(U32,Nat,word => U32.to_nat(Grid.start(grid,word)),U32.from_nat(U32.to_nat(piece)),piece,Arithmetic.native_roundtrip(piece)))

def reached_with(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,+index: U32,+inside: {Nat.is_le(U32.to_nat(index),total) == True{} : Bool},+choice: Bool,
  +test: {Nat.is_lt(U32.to_nat(Halo.piece_from(grid,index)),U32.to_nat(Grid.pieces_of(grid))) == choice : Bool}) ->
  {Nat.is_le(U32.to_nat(index),span_boundary(U32.to_nat(Halo.piece_from(grid,index)),layout,0n)) == True{} : Bool}:
  match choice:
    case True{}:
      Held.nat_le_right_moved(U32.to_nat(index),span_boundary(U32.to_nat(Halo.piece_from(grid,index)),layout,0n),U32.to_nat(Grid.start(grid,Halo.piece_from(grid,index))),
        starts_of(grid,layout,laid_along(limits,grid,layout,total,laid),Halo.piece_from(grid,index),test),after_of(limits,grid,layout,total,laid,index,inside,test))
    case False{}:
      +last = Order.antisymmetric(U32.to_nat(Halo.piece_from(grid,index)),U32.to_nat(Grid.pieces_of(grid)),
        Bounds.as_nat_le(Halo.piece_from(grid,index),Grid.pieces_of(grid),Laws.from_bounded(grid,index)),
        Order.not_less(U32.to_nat(Halo.piece_from(grid,index)),U32.to_nat(Grid.pieces_of(grid)),test))
      Held.nat_le_right_moved(U32.to_nat(index),span_boundary(U32.to_nat(Halo.piece_from(grid,index)),layout,0n),Laws.grid_cells(grid),
        along_end(grid,layout,0n,0n,laid_along(limits,grid,layout,total,laid),U32.to_nat(Halo.piece_from(grid,index)),last),
        Held.nat_le_right_moved(U32.to_nat(index),Laws.grid_cells(grid),total,Equal.sym(Nat,total,Laws.grid_cells(grid),laid_cells(limits,grid,layout,total,laid)),inside))

def reached_of(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>) -> ReachedAfter(grid,layout,total):
  +index => +inside => reached_with(limits,grid,layout,total,laid,index,inside,Nat.is_lt(U32.to_nat(Halo.piece_from(grid,index)),U32.to_nat(Grid.pieces_of(grid))),{==})

def maximum_inside(+left: U32,+right: U32,+total: Nat,+left_inside: {Nat.is_le(U32.to_nat(left),total) == True{} : Bool},
  +right_inside: {Nat.is_le(U32.to_nat(right),total) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(U32.max(left,right)),total) == True{} : Bool}:
  Held.nat_le_moved(U32.to_nat(U32.max(left,right)),Nat.max(U32.to_nat(left),U32.to_nat(right)),total,Extrema.maximum(left,right),
    NatExtrema.maximum_upper(U32.to_nat(left),U32.to_nat(right),total,left_inside,right_inside))

# ---- One halo ----

# The pieces [low, high) a task owns and the end of its halo.
def halo_low(+grid: Grid.Grid,+span: Tasks.Span,+reached: U32) -> U32:
  Halo.piece_from(grid,U32.max(span_start(span),reached))

def halo_high(+grid: Grid.Grid,+span: Tasks.Span,+reached: U32) -> U32:
  U32.max(halo_low(grid,span,reached),Halo.piece_from(grid,span_end(span)))

def halo_end(+grid: Grid.Grid,+span: Tasks.Span,+reached: U32) -> U32:
  Bool.pick(U32,U32.is_lt(halo_low(grid,span,reached),halo_high(grid,span,reached)),U32.min(Grid.start(grid,halo_low(grid,span,reached)),span_end(span)),span_end(span))

def halo_count(+grid: Grid.Grid,+span: Tasks.Span,+reached: U32) -> U32:
  (halo_end(grid,span,reached) - span_start(span) : U32)

type Between<-lower: U32,-value: U32,-upper: U32> is Data:
  Between{above: {Nat.is_le(U32.to_nat(lower),U32.to_nat(value)) == True{} : Bool},below: {Nat.is_le(U32.to_nat(value),U32.to_nat(upper)) == True{} : Bool}}

def end_between(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,+start: U32,+reached: U32,+finish: U32,+choice: Bool,
  +test: {U32.is_lt(Halo.piece_from(grid,U32.max(start,reached)),U32.max(Halo.piece_from(grid,U32.max(start,reached)),Halo.piece_from(grid,finish))) == choice : Bool},
  +ordered: {Nat.is_le(U32.to_nat(start),U32.to_nat(finish)) == True{} : Bool},+middle_inside: {Nat.is_le(U32.to_nat(U32.max(start,reached)),total) == True{} : Bool}) ->
  Between<start,Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,U32.max(start,reached))),finish),finish),finish>:
  match choice:
    case True{}:
      +middle = U32.max(start,reached)
      +low = Halo.piece_from(grid,middle)
      +high = U32.max(low,Halo.piece_from(grid,finish))
      +high_bounded = Bounds.as_nat_le(high,Grid.pieces_of(grid),
        Extrema.maximum_upper(low,Halo.piece_from(grid,finish),Grid.pieces_of(grid),Laws.from_bounded(grid,middle),Laws.from_bounded(grid,finish)))
      +inside = Bounds.nat_trans(U32.to_nat(low),U32.to_nat(high),U32.to_nat(Grid.pieces_of(grid)),Bounds.as_nat_lt(low,high,test),high_bounded)
      +after = Order.transitive(U32.to_nat(start),U32.to_nat(middle),U32.to_nat(Grid.start(grid,low)),Bounds.as_nat_le(start,middle,Extrema.maximum_left(start,reached)),
        after_of(limits,grid,layout,total,laid,middle,middle_inside,inside))
      Between{
        %Equal.sym(Nat,U32.to_nat(U32.min(Grid.start(grid,low),finish)),Nat.min(U32.to_nat(Grid.start(grid,low)),U32.to_nat(finish)),Extrema.minimum(Grid.start(grid,low),finish)) :
          {Nat.is_le(U32.to_nat(start),_) == True{} : Bool}
        NatExtrema.minimum_lower(U32.to_nat(Grid.start(grid,low)),U32.to_nat(finish),U32.to_nat(start),after,ordered),
        Bounds.as_nat_le(U32.min(Grid.start(grid,low),finish),finish,Extrema.minimum_right(Grid.start(grid,low),finish))}
    case False{}: Between{ordered,Order.reflexive(U32.to_nat(finish))}

# A halo of a reach inside the tensor ends inside it and can be allocated.
type HaloBounds<-start: U32,-count: U32,-total: Nat> is Data:
  HaloBounds{fits: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool},allocatable: {U32.is_le(count,1073741824) == True{} : Bool}}

def bounds_between(+start: U32,+finish: U32,+reach: U32,+total: Nat,+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},
  +small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},+finish_inside: {Nat.is_le(U32.to_nat(finish),total) == True{} : Bool},+between: Between<start,reach,finish>) ->
  HaloBounds<start,(reach - start : U32),total>:
  Between{+above,+below} = between
  +reach_inside = Order.transitive(U32.to_nat(reach),U32.to_nat(finish),total,below,finish_inside)
  +reach_small = Order.transitive(U32.to_nat(reach),total,U32.to_nat(limit),reach_inside,small)
  +exact = Arithmetic.subtract(reach,start,limit,Coordinates.native_le_transitive(limit,1073741824,2147483648,limit_ok,{==}),above,
    Order.transitive(Nat.sub(U32.to_nat(reach),U32.to_nat(start)),U32.to_nat(reach),U32.to_nat(limit),Order.sub_le(U32.to_nat(reach),U32.to_nat(start)),reach_small))
  +count_reach = Held.nat_le_moved(U32.to_nat((reach - start : U32)),Nat.sub(U32.to_nat(reach),U32.to_nat(start)),U32.to_nat(reach),exact,
    Order.sub_le(U32.to_nat(reach),U32.to_nat(start)))
  HaloBounds{
    Held.nat_le_moved(Nat.add(U32.to_nat(start),U32.to_nat((reach - start : U32))),U32.to_nat(reach),total,
      Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat((reach - start : U32))),Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(reach),U32.to_nat(start))),U32.to_nat(reach),
        Equal.cong(Nat,Nat,count => Nat.add(U32.to_nat(start),count),U32.to_nat((reach - start : U32)),Nat.sub(U32.to_nat(reach),U32.to_nat(start)),exact),
        Order.restore_shift(U32.to_nat(start),U32.to_nat(reach),above)),
      reach_inside),
    Coordinates.native_le_transitive((reach - start : U32),limit,1073741824,
      Held.u32_le((reach - start : U32),limit,Order.transitive(U32.to_nat((reach - start : U32)),U32.to_nat(reach),U32.to_nat(limit),count_reach,reach_small)),limit_ok)}

def halo_bounds(+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,+span: Tasks.Span,+reached: U32,
  +limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(span_start(span)),Nat.sub(U32.to_nat(span_end(span)),U32.to_nat(span_start(span)))),total) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(span_start(span)),U32.to_nat(span_end(span))) == True{} : Bool},
  +reached_inside: {Nat.is_le(U32.to_nat(reached),total) == True{} : Bool}) ->
  HaloBounds<span_start(span),halo_count(grid,span,reached),total>:
  +start = span_start(span)
  +finish = span_end(span)
  +finish_inside = Held.nat_le_moved(U32.to_nat(finish),Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),total,
    Equal.sym(Nat,Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),U32.to_nat(finish),Order.restore_shift(U32.to_nat(start),U32.to_nat(finish),ordered)),
    inside)
  bounds_between(start,finish,halo_end(grid,span,reached),total,limit,limit_ok,small,
    Held.nat_le_moved(U32.to_nat(finish),Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),total,
      Equal.sym(Nat,Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),U32.to_nat(finish),Order.restore_shift(U32.to_nat(start),U32.to_nat(finish),ordered)),
      inside),
    end_between(limits,grid,layout,total,laid,start,reached,finish,U32.is_lt(halo_low(grid,span,reached),halo_high(grid,span,reached)),{==},ordered,
      maximum_inside(start,reached,total,Order.transitive(U32.to_nat(start),U32.to_nat(finish),total,ordered,finish_inside),reached_inside)))

# ---- The first walk as a model ----

# The halos of the tasks: a buffer model, its cells and the pieces [low, high)
# of each task.
type HaloModel<-Element: Data> is Data:
  HaloLeaf{buffer: Buffer.Model<Element>,first: U32,count: U32,low: U32,high: U32,task_first: U32,task_count: U32}
  HaloFork{left: HaloModel<Element>,right: HaloModel<Element>}

def pack_halos(~Element: Data,halos: HaloModel<Element>) -> Partition.Tree<&1,Halo.Halo<Element>,Unit>:
  match halos:
    case HaloLeaf{buffer,first,count,low,high,task_first,task_count}: Partition.Leaf{Halo.Halo{Buffer.pack(~Element,buffer),first,count,low,high},task_first,task_count,Unit{}}
    case HaloFork{left,right}: Partition.Fork{pack_halos(~Element,left),pack_halos(~Element,right)}

# The input pieces after the walk and the halos it made.
type Walked<-Element: Data> is Data:
  Walked{holding: Held.Holding<Element>,halos: HaloModel<Element>}

def walked_holding(~Element: Data,walked: Walked<Element>) -> Held.Holding<Element>:
  Walked{holding,halos} = walked
  holding

def walked_halos(~Element: Data,walked: Walked<Element>) -> HaloModel<Element>:
  Walked{holding,halos} = walked
  halos

def walked_eta(~Element: Data,+walked: Walked<Element>) -> {walked == Walked{walked_holding(~Element,walked),walked_halos(~Element,walked)} : Walked<Element>}:
  match walked:
    case Walked{holding,halos}: {==}

def halo_copy(~Element: Data,+grid: Grid.Grid,+fill: Element,+span: Tasks.Span,+holding: Held.Holding<Element>,+reached: U32) ->
  Move.Holding<Element> & Storage.Buffer<Element>:
  Move.copied(~Element,Held.pack(~Element,holding),span_start(span),halo_count(grid,span,reached),fill)

def walked_leaf(~Element: Data,+grid: Grid.Grid,+fill: Element,+task_first: U32,+task_count: U32,+span: Tasks.Span,+holding: Held.Holding<Element>,+reached: U32) -> Walked<Element>:
  Walked{Held.copied_holding(~Element,holding,span_start(span),halo_count(grid,span,reached),fill),
    HaloLeaf{Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,span,holding,reached))),
      span_start(span),halo_count(grid,span,reached),halo_low(grid,span,reached),halo_high(grid,span,reached),task_first,task_count}}

def reach_after(~Task: Data,~reach: Halo.Reach(~Task),+task: Task,tasks: Pieces.Ranges,+reached: U32) -> U32:
  match tasks:
    case Pieces.Range{+first,+count}: U32.max(reached,span_end(reach(task,first,count)))
    case Pieces.Split{+left,right}: reach_after(~Task,~reach,task,right,reach_after(~Task,~reach,task,left,reached))

# The reach after tasks stays inside the tensor.
def reach_after_inside(~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+total: Nat,+tasks: Pieces.Ranges,+reaches: Reaches(~Task,~reach,task,tasks,total),
  +reached: U32,+bounded: {Nat.is_le(U32.to_nat(reached),total) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(reach_after(~Task,~reach,task,tasks,reached)),total) == True{} : Bool}:
  match tasks:
    case Pieces.Range{+first,+count}:
      +span = reach(task,first,count)
      maximum_inside(reached,span_end(span),total,bounded,
        Held.nat_le_moved(U32.to_nat(span_end(span)),Nat.add(U32.to_nat(span_start(span)),Nat.sub(U32.to_nat(span_end(span)),U32.to_nat(span_start(span)))),total,
          Equal.sym(Nat,Nat.add(U32.to_nat(span_start(span)),Nat.sub(U32.to_nat(span_end(span)),U32.to_nat(span_start(span)))),U32.to_nat(span_end(span)),
            Order.restore_shift(U32.to_nat(span_start(span)),U32.to_nat(span_end(span)),Held.second_of({Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,first,count))),Nat.sub(U32.to_nat(span_end(reach(task,first,count))),U32.to_nat(span_start(reach(task,first,count))))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(span_start(reach(task,first,count))),U32.to_nat(span_end(reach(task,first,count)))) == True{} : Bool},reaches))),
          Held.first_of({Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,first,count))),Nat.sub(U32.to_nat(span_end(reach(task,first,count))),U32.to_nat(span_start(reach(task,first,count))))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(span_start(reach(task,first,count))),U32.to_nat(span_end(reach(task,first,count)))) == True{} : Bool},reaches)))
    case Pieces.Split{+left,+right}:
      reach_after_inside(~Task,~reach,task,total,right,Held.second_of(Reaches(~Task,~reach,task,left,total),Reaches(~Task,~reach,task,right,total),reaches),reach_after(~Task,~reach,task,left,reached),
        reach_after_inside(~Task,~reach,task,total,left,Held.first_of(Reaches(~Task,~reach,task,left,total),Reaches(~Task,~reach,task,right,total),reaches),reached,bounded))

def walked_join(~Element: Data,+left: HaloModel<Element>,right: Walked<Element>) -> Walked<Element>:
  Walked{holding,halos} = right
  Walked{holding,HaloFork{left,halos}}

def walk_model(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,tasks: Pieces.Ranges,+holding: Held.Holding<Element>,
  +reached: U32) -> Walked<Element>:
  match tasks:
    case Pieces.Range{+first,+count}: walked_leaf(~Element,grid,fill,first,count,reach(task,first,count),holding,reached)
    case Pieces.Split{+left,+right}:
      +left_walk = walk_model(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      walked_join(~Element,walked_halos(~Element,left_walk),
        walk_model(~Element,~Task,~reach,task,grid,fill,right,walked_holding(~Element,left_walk),reach_after(~Task,~reach,task,left,reached)))

def pack_walked(~Element: Data,+walked: Walked<Element>,+reached: U32) -> Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>:
  (Halo.Reading{Held.pack(~Element,walked_holding(~Element,walked)),reached},pack_halos(~Element,walked_halos(~Element,walked)))

# ---- The first walk refines its model ----

def halo_build(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,tasks: Pieces.Ranges,state: Halo.Reading<Element>) ->
  Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>:
  Pieces.build(~Halo.Halo<Element>,~Halo.Reading<Element>,~Halo.Reaching<Element,Task>,~Halo.halo_task(~Element,~Task,~reach),tasks,Halo.Reaching{task,grid,fill},state)

def spanned_packed(~Element: Data,+grid: Grid.Grid,+fill: Element,+task_first: U32,+task_count: U32,+span: Tasks.Span,+holding: Held.Holding<Element>,+reached: U32) ->
  {Halo.halo_spanned(~Element,grid,fill,task_first,task_count,span,Halo.Reading{Held.pack(~Element,holding),reached}) ==
    pack_walked(~Element,walked_leaf(~Element,grid,fill,task_first,task_count,span,holding,reached),U32.max(reached,span_end(span))) :
    Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}:
  match span:
    case Tasks.Span{+start,+length}:
      +whole = span_of(start,length)
      +count = halo_count(grid,whole,reached)
      +low = halo_low(grid,whole,reached)
      +high = halo_high(grid,whole,reached)
      +reached_after = U32.max(reached,span_end(whole))
      %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached),
        (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached)),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached))),
        Held.pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached))) :
        {Halo.halo_leaf(~Element,task_first,task_count,start,count,low,high,reached_after,_) ==
          pack_walked(~Element,walked_leaf(~Element,grid,fill,task_first,task_count,whole,holding,reached),reached_after) : Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      %Equal.sym(Move.Holding<Element>,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached)),Held.pack(~Element,Held.copied_holding(~Element,holding,start,count,fill)),
        Held.copied_held(~Element,holding,start,count,fill)) :
        {(Halo.Reading{_,reached_after},Partition.Leaf{Halo.Halo{Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached)),start,count,low,high},task_first,task_count,Unit{}}) ==
          pack_walked(~Element,walked_leaf(~Element,grid,fill,task_first,task_count,whole,holding,reached),reached_after) : Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      %Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,whole,holding,reached))) :
        {(Halo.Reading{Held.pack(~Element,Held.copied_holding(~Element,holding,start,count,fill)),reached_after},
          Partition.Leaf{Halo.Halo{_,start,count,low,high},task_first,task_count,Unit{}}) ==
          pack_walked(~Element,walked_leaf(~Element,grid,fill,task_first,task_count,whole,holding,reached),reached_after) : Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      {==}

def walked_parts(~Element: Data,+walked: Walked<Element>,+reached: U32) ->
  {pack_walked(~Element,walked,reached) == (Halo.Reading{Held.pack(~Element,walked_holding(~Element,walked)),reached},pack_halos(~Element,walked_halos(~Element,walked))) :
    Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}:
  {==}

# The first walk from a packed holding is the pack of its model.
def walk_packed(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,+tasks: Pieces.Ranges,+holding: Held.Holding<Element>,
  +reached: U32) ->
  {halo_build(~Element,~Task,~reach,task,grid,fill,tasks,Halo.Reading{Held.pack(~Element,holding),reached}) ==
    pack_walked(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,reached),reach_after(~Task,~reach,task,tasks,reached)) :
    Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}:
  match tasks:
    case Pieces.Range{+first,+count}: spanned_packed(~Element,grid,fill,first,count,reach(task,first,count),holding,reached)
    case Pieces.Split{+left,+right}:
      +left_walk = walk_model(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      +left_reached = reach_after(~Task,~reach,task,left,reached)
      +left_holding = walked_holding(~Element,left_walk)
      +left_halos = walked_halos(~Element,left_walk)
      +right_walk = walk_model(~Element,~Task,~reach,task,grid,fill,right,left_holding,left_reached)
      +right_reached = reach_after(~Task,~reach,task,right,left_reached)
      %Equal.sym(Walked<Element>,right_walk,Walked{walked_holding(~Element,right_walk),walked_halos(~Element,right_walk)},walked_eta(~Element,right_walk)) :
        {halo_build(~Element,~Task,~reach,task,grid,fill,Pieces.Split{left,right},Halo.Reading{Held.pack(~Element,holding),reached}) ==
          pack_walked(~Element,walked_join(~Element,left_halos,_),right_reached) : Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      %Equal.sym(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,
        halo_build(~Element,~Task,~reach,task,grid,fill,left,Halo.Reading{Held.pack(~Element,holding),reached}),
        (Halo.Reading{Held.pack(~Element,left_holding),left_reached},pack_halos(~Element,left_halos)),
        Equal.trans(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,
          halo_build(~Element,~Task,~reach,task,grid,fill,left,Halo.Reading{Held.pack(~Element,holding),reached}),pack_walked(~Element,left_walk,left_reached),
          (Halo.Reading{Held.pack(~Element,left_holding),left_reached},pack_halos(~Element,left_halos)),
          walk_packed(~Element,~Task,~reach,task,grid,fill,left,holding,reached),walked_parts(~Element,left_walk,left_reached))) :
        {Storage.bind_pair(~Halo.Reading<Element>,~Partition.Tree<&1,Halo.Halo<Element>,Unit>,~(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>),_,
          state => left_tree => Storage.bind_pair(~Halo.Reading<Element>,~Partition.Tree<&1,Halo.Halo<Element>,Unit>,~(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>),
            halo_build(~Element,~Task,~reach,task,grid,fill,right,state),state => right_tree => (state,Partition.Fork{left_tree,right_tree}))) ==
          (Halo.Reading{Held.pack(~Element,walked_holding(~Element,right_walk)),right_reached},
            Partition.Fork{pack_halos(~Element,left_halos),pack_halos(~Element,walked_halos(~Element,right_walk))}) :
          Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      %Equal.sym(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,
        halo_build(~Element,~Task,~reach,task,grid,fill,right,Halo.Reading{Held.pack(~Element,left_holding),left_reached}),
        (Halo.Reading{Held.pack(~Element,walked_holding(~Element,right_walk)),right_reached},pack_halos(~Element,walked_halos(~Element,right_walk))),
        Equal.trans(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,
          halo_build(~Element,~Task,~reach,task,grid,fill,right,Halo.Reading{Held.pack(~Element,left_holding),left_reached}),pack_walked(~Element,right_walk,right_reached),
          (Halo.Reading{Held.pack(~Element,walked_holding(~Element,right_walk)),right_reached},pack_halos(~Element,walked_halos(~Element,right_walk))),
          walk_packed(~Element,~Task,~reach,task,grid,fill,right,left_holding,left_reached),walked_parts(~Element,right_walk,right_reached))) :
        {Storage.bind_pair(~Halo.Reading<Element>,~Partition.Tree<&1,Halo.Halo<Element>,Unit>,~(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>),_,
          state => right_tree => (state,Partition.Fork{pack_halos(~Element,left_halos),right_tree})) ==
          (Halo.Reading{Held.pack(~Element,walked_holding(~Element,right_walk)),right_reached},
            Partition.Fork{pack_halos(~Element,left_halos),pack_halos(~Element,walked_halos(~Element,right_walk))}) :
          Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>}
      {==}

# ---- The first walk keeps the input pieces ----

# A holding of the input pieces: they cover the tensor with its values in the
# order of its layout.
type HeldFacts<-Element: Data,-Source: Data,-value: Source -> U32 -> Element,-source: Source,-holding: Held.Holding<Element>,-total: Nat,-layout: Held.Spans> is Data:
  HeldFacts{tiled: Held.Tiled(~Element,holding,0n,total),values: Held.HoldingValues(~Element,~Source,~value,source,holding),
    ordered: {Held.spans(~Element,Held.order(~Element,holding)) == layout : Held.Spans}}

def copy_facts(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,+source: Source,+holding: Held.Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +total: Nat,+layout: Held.Spans,+facts: HeldFacts<Element,Source,value,source,holding,total,layout>) ->
  HeldFacts<Element,Source,value,source,Held.copied_holding(~Element,holding,first,count,fill),total,layout>:
  HeldFacts{+tiled,+values,+ordered} = facts
  +fresh = Buffer.tree(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))
  HeldFacts{Held.run_tiled(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh,0n,total,tiled),
    Held.run_values(~Element,~Source,~value,source,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh,values),
    Equal.trans(Held.Spans,Held.spans(~Element,Held.order(~Element,Held.copied_holding(~Element,holding,first,count,fill))),Held.spans(~Element,Held.order(~Element,holding)),layout,
      Equal.cong(Held.Stack<Element>,Held.Spans,Held.spans(~Element),Held.order(~Element,Held.copied_holding(~Element,holding,first,count,fill)),Held.order(~Element,holding),
        Held.run_order(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh)),ordered)}

def join_holding(~Element: Data,+left: HaloModel<Element>,+right: Walked<Element>) ->
  {walked_holding(~Element,walked_join(~Element,left,right)) == walked_holding(~Element,right) : Held.Holding<Element>}:
  match right:
    case Walked{holding,halos}: {==}

def join_halos(~Element: Data,+left: HaloModel<Element>,+right: Walked<Element>) ->
  {walked_halos(~Element,walked_join(~Element,left,right)) == HaloFork{left,walked_halos(~Element,right)} : HaloModel<Element>}:
  match right:
    case Walked{holding,halos}: {==}

# The holding after the first walk still holds the input pieces.
def walk_facts(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+source: Source,+task: Task,+grid: Grid.Grid,+fill: Element,
  +tasks: Pieces.Ranges,+holding: Held.Holding<Element>,+reached: U32,+total: Nat,+layout: Held.Spans,+facts: HeldFacts<Element,Source,value,source,holding,total,layout>) ->
  HeldFacts<Element,Source,value,source,walked_holding(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,reached)),total,layout>:
  match tasks:
    case Pieces.Range{+first,+count}:
      copy_facts(~Element,~Source,~value,source,holding,span_start(reach(task,first,count)),halo_count(grid,reach(task,first,count),reached),fill,total,layout,facts)
    case Pieces.Split{+left,+right}:
      +left_walk = walk_model(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      +right_walk = walk_model(~Element,~Task,~reach,task,grid,fill,right,walked_holding(~Element,left_walk),reach_after(~Task,~reach,task,left,reached))
      %Equal.sym(Held.Holding<Element>,walked_holding(~Element,walked_join(~Element,walked_halos(~Element,left_walk),right_walk)),walked_holding(~Element,right_walk),
        join_holding(~Element,walked_halos(~Element,left_walk),right_walk)) :
        HeldFacts<Element,Source,value,source,_,total,layout>
      walk_facts(~Element,~Source,~value,~Task,~reach,source,task,grid,fill,right,walked_holding(~Element,left_walk),reach_after(~Task,~reach,task,left,reached),total,layout,
        walk_facts(~Element,~Source,~value,~Task,~reach,source,task,grid,fill,left,holding,reached,total,layout,facts))

# ---- The second walk as a model ----

def skipped_packed(~Element: Data,+count: Nat,+stack: Held.Stack<Element>) ->
  {Halo.skipped(~Element,count,Held.pack_stack(~Element,stack)) == Held.pack_stack(~Element,Held.dropped(~Element,count,stack)) : Move.Stack<Element>}:
  match count stack:
    case 1n+ +rest Held.Held{+buffer,+first,+pieces,+below}: skipped_packed(~Element,rest,below)
    case 0n stack: {==}
    case 1n+rest Held.Bottom{}: {==}

def handed_packed(~Element: Data,+count: Nat,+stack: Held.Stack<Element>,+own: Held.Stack<Element>) ->
  {Halo.handed(~Element,count,Held.pack_stack(~Element,stack),Held.pack_stack(~Element,own)) ==
    Halo.Handed{Held.pack_stack(~Element,Held.onto(~Element,Held.taken(~Element,count,stack),own)),Held.pack_stack(~Element,Held.dropped(~Element,count,stack))} : Halo.Handed<Element>}:
  match count stack:
    case 1n+ +rest Held.Held{+buffer,+first,+pieces,+below}: handed_packed(~Element,rest,below,Held.Held{buffer,first,pieces,own})
    case 0n stack: {==}
    case 1n+rest Held.Bottom{}: {==}

# A task's holding: its halo, when it has cells, before its own pieces.
def window_holding(~Element: Data,empty: Bool,+halo: Buffer.Model<Element>,+first: U32,+count: U32,+own: Held.Stack<Element>) -> Held.Holding<Element>:
  match empty:
    case True{}: Held.Holding{Held.Bottom{},own}
    case False{}: Held.Holding{Held.Bottom{},Held.Held{halo,first,count,own}}

def window_packed(~Element: Data,+empty: Bool,+halo: Buffer.Model<Element>,+first: U32,+count: U32,+own: Held.Stack<Element>) ->
  {Halo.window_of(~Element,empty,Buffer.pack(~Element,halo),first,count,Held.pack_stack(~Element,own)) == Held.pack(~Element,window_holding(~Element,empty,halo,first,count,own)) :
    Move.Holding<Element>}:
  match empty:
    case True{}: {==}
    case False{}: {==}

type WindowModel<-Element: Data> is Data:
  WindowLeaf{holding: Held.Holding<Element>,task_first: U32,task_count: U32}
  WindowFork{left: WindowModel<Element>,right: WindowModel<Element>}

def pack_windows(~Element: Data,windows: WindowModel<Element>) -> Partition.Tree<&1,Move.Holding<Element>,Unit>:
  match windows:
    case WindowLeaf{holding,task_first,task_count}: Partition.Leaf{Held.pack(~Element,holding),task_first,task_count,Unit{}}
    case WindowFork{left,right}: Partition.Fork{pack_windows(~Element,left),pack_windows(~Element,right)}

# The windows of the tasks so far, the pieces not yet handed and the index of
# the first of them.
type Handled<-Element: Data> is Data:
  Handled{windows: WindowModel<Element>,stack: Held.Stack<Element>,next: U32}

def handled_windows(~Element: Data,handled: Handled<Element>) -> WindowModel<Element>:
  Handled{windows,stack,next} = handled
  windows

def handled_stack(~Element: Data,handled: Handled<Element>) -> Held.Stack<Element>:
  Handled{windows,stack,next} = handled
  stack

def handled_next(~Element: Data,handled: Handled<Element>) -> U32:
  Handled{windows,stack,next} = handled
  next

def handled_eta(~Element: Data,+handled: Handled<Element>) ->
  {handled == Handled{handled_windows(~Element,handled),handled_stack(~Element,handled),handled_next(~Element,handled)} : Handled<Element>}:
  match handled:
    case Handled{windows,stack,next}: {==}

# The window of a task that takes the pieces [start, finish) of the stack from
# index next on.
def handled_at(~Element: Data,+halo: Buffer.Model<Element>,+first: U32,+count: U32,+task_first: U32,+task_count: U32,+stack: Held.Stack<Element>,+next: U32,
  +start: U32,+finish: U32) -> Handled<Element>:
  +rest = Held.dropped(~Element,U32.to_nat((start - next : U32)),stack)
  +keep = U32.to_nat((finish - start : U32))
  Handled{WindowLeaf{window_holding(~Element,U32.is_zero(count),halo,first,count,Held.onto(~Element,Held.onto(~Element,Held.taken(~Element,keep,rest),Held.Bottom{}),Held.Bottom{})),
    task_first,task_count},Held.dropped(~Element,keep,rest),finish}

# The pieces a task takes: from the index max(low, next) to max(high, it).
def handled_leaf(~Element: Data,+halo: Buffer.Model<Element>,+first: U32,+count: U32,+low: U32,+high: U32,+task_first: U32,+task_count: U32,
  +stack: Held.Stack<Element>,+next: U32) -> Handled<Element>:
  handled_at(~Element,halo,first,count,task_first,task_count,stack,next,U32.max(low,next),U32.max(high,U32.max(low,next)))

def handled_join(~Element: Data,+left: WindowModel<Element>,right: Handled<Element>) -> Handled<Element>:
  Handled{windows,stack,next} = right
  Handled{WindowFork{left,windows},stack,next}

def handled_model(~Element: Data,halos: HaloModel<Element>,+stack: Held.Stack<Element>,+next: U32) -> Handled<Element>:
  match halos:
    case HaloLeaf{+buffer,+first,+count,+low,+high,+task_first,+task_count}: handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)
    case HaloFork{left,right}:
      +left_handled = handled_model(~Element,left,stack,next)
      handled_join(~Element,handled_windows(~Element,left_handled),
        handled_model(~Element,right,handled_stack(~Element,left_handled),handled_next(~Element,left_handled)))

def pack_handled(~Element: Data,+handled: Handled<Element>) -> Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>:
  (pack_windows(~Element,handled_windows(~Element,handled)),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled)),handled_next(~Element,handled)})

# ---- The second walk refines its model ----

def handled_parts(~Element: Data,+handled: Handled<Element>) ->
  {pack_handled(~Element,handled) == (pack_windows(~Element,handled_windows(~Element,handled)),
    Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled)),handled_next(~Element,handled)}) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}:
  {==}

# The leaf of the second walk over a packed halo and a packed stack is the
# pack of its model.
def leaf_packed(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+low: U32,+high: U32,+task_first: U32,+task_count: U32,
  +stack: Held.Stack<Element>,+next: U32) ->
  {Halo.window_leaf(~Element,Unit{},task_first,task_count,Halo.Halo{Buffer.pack(~Element,buffer),first,count,low,high},Halo.Handing{Held.pack_stack(~Element,stack),next}) ==
    pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}:
  +start = U32.max(low,next)
  +finish = U32.max(high,start)
  +skip = U32.to_nat((start - next : U32))
  +rest = Held.dropped(~Element,skip,stack)
  +keep = U32.to_nat((finish - start : U32))
  +own = Held.onto(~Element,Held.taken(~Element,keep,rest),Held.Bottom{})
  %Equal.sym(Move.Stack<Element>,Halo.skipped(~Element,skip,Held.pack_stack(~Element,stack)),Held.pack_stack(~Element,rest),skipped_packed(~Element,skip,stack)) :
    {Halo.window_handed(~Element,task_first,task_count,Buffer.pack(~Element,buffer),first,count,finish,Halo.handed(~Element,keep,_,Move.Bottom{})) == pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}
  %Equal.sym(Halo.Handed<Element>,Halo.handed(~Element,keep,Held.pack_stack(~Element,rest),Held.pack_stack(~Element,Held.Bottom{})),
    Halo.Handed{Held.pack_stack(~Element,own),Held.pack_stack(~Element,Held.dropped(~Element,keep,rest))},handed_packed(~Element,keep,rest,Held.Bottom{})) :
    {Halo.window_handed(~Element,task_first,task_count,Buffer.pack(~Element,buffer),first,count,finish,_) == pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}
  %Equal.sym(Move.Stack<Element>,Move.onto_ahead(~Element,Held.pack_stack(~Element,own),Held.pack_stack(~Element,Held.Bottom{})),
    Held.pack_stack(~Element,Held.onto(~Element,own,Held.Bottom{})),Held.onto_ahead_packed(~Element,own,Held.Bottom{})) :
    {(Partition.Leaf{Halo.window_of(~Element,U32.is_zero(count),Buffer.pack(~Element,buffer),first,count,_),task_first,task_count,Unit{}},
      Halo.Handing{Held.pack_stack(~Element,Held.dropped(~Element,keep,rest)),finish}) == pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}
  %Equal.sym(Move.Holding<Element>,
    Halo.window_of(~Element,U32.is_zero(count),Buffer.pack(~Element,buffer),first,count,Held.pack_stack(~Element,Held.onto(~Element,own,Held.Bottom{}))),
    Held.pack(~Element,window_holding(~Element,U32.is_zero(count),buffer,first,count,Held.onto(~Element,own,Held.Bottom{}))),
    window_packed(~Element,U32.is_zero(count),buffer,first,count,Held.onto(~Element,own,Held.Bottom{}))) :
    {(Partition.Leaf{_,task_first,task_count,Unit{}},Halo.Handing{Held.pack_stack(~Element,Held.dropped(~Element,keep,rest)),finish}) == pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) : Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>}
  {==}

# The trees of windows waiting on the left of the second walk.
def pack_window_lefts(~Element: Data,models: List<&2,WindowModel<Element>>) -> Pieces.Lefts<Move.Holding<Element>>:
  match models:
    case Nil{}: Pieces.NoLefts{}
    case Con{model,rest}: Pieces.Left{pack_windows(~Element,model),pack_window_lefts(~Element,rest)}

# The second walk over packed halos and a packed stack keeps the windows
# waiting on its left and gives the pack of its model.
def handled_walked(~Element: Data,halos: HaloModel<Element>,+models: List<&2,WindowModel<Element>>,+stack: Held.Stack<Element>,+next: U32) ->
  {Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,halos),Pieces.Walking{pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,stack),next}}) ==
    Pieces.Walked{pack_windows(~Element,handled_windows(~Element,handled_model(~Element,halos,stack,next))),pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled_model(~Element,halos,stack,next))),handled_next(~Element,handled_model(~Element,halos,stack,next))}} : Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>}:
  match halos:
    case HaloLeaf{+buffer,+first,+count,+low,+high,+task_first,+task_count}:
      %Equal.sym(Partition.Tree<&1,Move.Holding<Element>,Unit> & Halo.Handing<Element>,
        Halo.window_leaf(~Element,Unit{},task_first,task_count,Halo.Halo{Buffer.pack(~Element,buffer),first,count,low,high},Halo.Handing{Held.pack_stack(~Element,stack),next}),
        pack_handled(~Element,handled_leaf(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)),
        leaf_packed(~Element,buffer,first,count,low,high,task_first,task_count,stack,next)) :
        {Pieces.leaf_walked(~Move.Holding<Element>,~Halo.Handing<Element>,pack_window_lefts(~Element,models),_) ==
          Pieces.Walked{pack_windows(~Element,handled_windows(~Element,handled_model(~Element,HaloLeaf{buffer,first,count,low,high,task_first,task_count},stack,next))),pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled_model(~Element,HaloLeaf{buffer,first,count,low,high,task_first,task_count},stack,next))),handled_next(~Element,handled_model(~Element,HaloLeaf{buffer,first,count,low,high,task_first,task_count},stack,next))}} : Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>}
      {==}
    case HaloFork{+left,+right}:
      +left_handled = handled_model(~Element,left,stack,next)
      +left_windows = handled_windows(~Element,left_handled)
      +left_stack = handled_stack(~Element,left_handled)
      +left_next = handled_next(~Element,left_handled)
      +right_handled = handled_model(~Element,right,left_stack,left_next)
      %Equal.sym(Handled<Element>,right_handled,
        Handled{handled_windows(~Element,right_handled),handled_stack(~Element,right_handled),handled_next(~Element,right_handled)},handled_eta(~Element,right_handled)) :
        {Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,HaloFork{left,right}),Pieces.Walking{pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,stack),next}}) ==
          Pieces.Walked{pack_windows(~Element,handled_windows(~Element,handled_join(~Element,left_windows,_))),pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled_join(~Element,left_windows,_))),handled_next(~Element,handled_join(~Element,left_windows,_))}} : Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>}
      %Equal.sym(Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>,Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,left),Pieces.Walking{pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,stack),next}}),
        Pieces.Walked{pack_windows(~Element,handled_windows(~Element,left_handled)),pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,left_handled)),handled_next(~Element,left_handled)}},handled_walked(~Element,left,models,stack,next)) :
        {Pieces.joined(~Move.Holding<Element>,~Halo.Handing<Element>,Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},
          pack_halos(~Element,right),Pieces.pushed(~Move.Holding<Element>,~Halo.Handing<Element>,_))) == Pieces.Walked{Partition.Fork{pack_windows(~Element,left_windows),pack_windows(~Element,handled_windows(~Element,right_handled))},pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,right_handled)),handled_next(~Element,right_handled)}} : Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>}
      %Equal.sym(Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>,Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,right),Pieces.Walking{pack_window_lefts(~Element,Con{left_windows,models}),Halo.Handing{Held.pack_stack(~Element,left_stack),left_next}}),
        Pieces.Walked{pack_windows(~Element,handled_windows(~Element,right_handled)),pack_window_lefts(~Element,Con{left_windows,models}),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,right_handled)),handled_next(~Element,right_handled)}},handled_walked(~Element,right,Con{left_windows,models},left_stack,left_next)) :
        {Pieces.joined(~Move.Holding<Element>,~Halo.Handing<Element>,_) == Pieces.Walked{Partition.Fork{pack_windows(~Element,left_windows),pack_windows(~Element,handled_windows(~Element,right_handled))},pack_window_lefts(~Element,models),Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,right_handled)),handled_next(~Element,right_handled)}} : Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>}
      {==}

# ---- Halo.windows refines its model ----

# The first walk keeps the input pieces in their order.
def walk_order(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,+tasks: Pieces.Ranges,+holding: Held.Holding<Element>,
  +reached: U32) ->
  {Held.order(~Element,walked_holding(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,reached))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match tasks:
    case Pieces.Range{+first,+count}:
      +span = reach(task,first,count)
      +halo = halo_count(grid,span,reached)
      Held.run_order(~Element,U32.to_nat(halo),U32.is_zero(halo),halo,span_start(span),1,False{},0,holding,
        Buffer.tree(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(halo),fill,U32.to_nat(halo)))))
    case Pieces.Split{+left,+right}:
      +left_walk = walk_model(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      +right_walk = walk_model(~Element,~Task,~reach,task,grid,fill,right,walked_holding(~Element,left_walk),reach_after(~Task,~reach,task,left,reached))
      %Equal.sym(Held.Holding<Element>,walked_holding(~Element,walked_join(~Element,walked_halos(~Element,left_walk),right_walk)),walked_holding(~Element,right_walk),
        join_holding(~Element,walked_halos(~Element,left_walk),right_walk)) :
        {Held.order(~Element,_) == Held.order(~Element,holding) : Held.Stack<Element>}
      Equal.trans(Held.Stack<Element>,Held.order(~Element,walked_holding(~Element,right_walk)),Held.order(~Element,walked_holding(~Element,left_walk)),Held.order(~Element,holding),
        walk_order(~Element,~Task,~reach,task,grid,fill,right,walked_holding(~Element,left_walk),reach_after(~Task,~reach,task,left,reached)),
        walk_order(~Element,~Task,~reach,task,grid,fill,left,holding,reached))

# The windows the tasks hold are the pack of the model of the two walks.
def windows_model(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,+tasks: Pieces.Ranges,+holding: Held.Holding<Element>) ->
  WindowModel<Element>:
  +walked = walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,0)
  handled_windows(~Element,handled_model(~Element,walked_halos(~Element,walked),Held.order(~Element,walked_holding(~Element,walked)),0))

def windows_packed(~Element: Data,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+fill: Element,+tasks: Pieces.Ranges,pieces: Pieces.Pieces(~Element),
  +holding: Held.Holding<Element>,+packed: {Move.held(~Element,pieces) == Held.pack(~Element,holding) : Move.Holding<Element>},
  +nonempty: {Held.spanned(Held.spans(~Element,Held.order(~Element,holding))) == True{} : Bool}) ->
  {Halo.windows(~Element,~Task,~reach,task,grid,fill,tasks,pieces) == pack_windows(~Element,windows_model(~Element,~Task,~reach,task,grid,fill,tasks,holding)) :
    Partition.Tree<&1,Move.Holding<Element>,Unit>}:
  +walked = walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,0)
  +walked_reached = reach_after(~Task,~reach,task,tasks,0)
  +result = windows_model(~Element,~Task,~reach,task,grid,fill,tasks,holding)
  %Equal.sym(Move.Holding<Element>,Move.held(~Element,pieces),Held.pack(~Element,holding),packed) :
    {Halo.windows_handed(~Element,Halo.halos_of(~Element,halo_build(~Element,~Task,~reach,task,grid,fill,tasks,Halo.Reading{_,0}),fill)) == pack_windows(~Element,result) :
      Partition.Tree<&1,Move.Holding<Element>,Unit>}
  %Equal.sym(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,halo_build(~Element,~Task,~reach,task,grid,fill,tasks,Halo.Reading{Held.pack(~Element,holding),0}),
    (Halo.Reading{Held.pack(~Element,walked_holding(~Element,walked)),walked_reached},pack_halos(~Element,walked_halos(~Element,walked))),
    Equal.trans(Halo.Reading<Element> & Partition.Tree<&1,Halo.Halo<Element>,Unit>,halo_build(~Element,~Task,~reach,task,grid,fill,tasks,Halo.Reading{Held.pack(~Element,holding),0}),
      pack_walked(~Element,walked,walked_reached),(Halo.Reading{Held.pack(~Element,walked_holding(~Element,walked)),walked_reached},pack_halos(~Element,walked_halos(~Element,walked))),
      walk_packed(~Element,~Task,~reach,task,grid,fill,tasks,holding,0),walked_parts(~Element,walked,walked_reached))) :
    {Halo.windows_handed(~Element,Halo.halos_of(~Element,_,fill)) == pack_windows(~Element,result) : Partition.Tree<&1,Move.Holding<Element>,Unit>}
  %Equal.sym(Move.Stack<Element>,Move.stacked(~Element,Move.released(~Element,Held.pack(~Element,walked_holding(~Element,walked)),fill),Move.Bottom{}),
    Held.pack_stack(~Element,Held.order(~Element,walked_holding(~Element,walked))),
    Held.released_stacked(~Element,walked_holding(~Element,walked),fill,
      Equal.trans(Bool,Held.spanned(Held.spans(~Element,Held.order(~Element,walked_holding(~Element,walked)))),Held.spanned(Held.spans(~Element,Held.order(~Element,holding))),True{},
        Equal.cong(Held.Stack<Element>,Bool,stack => Held.spanned(Held.spans(~Element,stack)),Held.order(~Element,walked_holding(~Element,walked)),Held.order(~Element,holding),
          walk_order(~Element,~Task,~reach,task,grid,fill,tasks,holding,0)),nonempty))) :
    {Pair.fst(Partition.Tree<&1,Move.Holding<Element>,Unit>,Halo.Handing<Element>,Pieces.walk(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,walked_halos(~Element,walked)),Halo.Handing{_,0})) ==
      pack_windows(~Element,result) : Partition.Tree<&1,Move.Holding<Element>,Unit>}
  %Equal.sym(Pieces.Walked<Move.Holding<Element>,Halo.Handing<Element>>,
    Pieces.walked(~Halo.Halo<Element>,~Move.Holding<Element>,~Halo.Handing<Element>,~Unit,~Halo.window_leaf(~Element),Unit{},pack_halos(~Element,walked_halos(~Element,walked)),
      Pieces.Walking{pack_window_lefts(~Element,Nil{}),Halo.Handing{Held.pack_stack(~Element,Held.order(~Element,walked_holding(~Element,walked))),0}}),
    Pieces.Walked{pack_windows(~Element,handled_windows(~Element,handled_model(~Element,walked_halos(~Element,walked),Held.order(~Element,walked_holding(~Element,walked)),0))),pack_window_lefts(~Element,Nil{}),
      Halo.Handing{Held.pack_stack(~Element,handled_stack(~Element,handled_model(~Element,walked_halos(~Element,walked),Held.order(~Element,walked_holding(~Element,walked)),0))),handled_next(~Element,handled_model(~Element,walked_halos(~Element,walked),Held.order(~Element,walked_holding(~Element,walked)),0))}},
    handled_walked(~Element,walked_halos(~Element,walked),Nil{},Held.order(~Element,walked_holding(~Element,walked)),0)) :
    {Pair.fst(Partition.Tree<&1,Move.Holding<Element>,Unit>,Halo.Handing<Element>,Pieces.walk_result(~Move.Holding<Element>,~Halo.Handing<Element>,_)) ==
      pack_windows(~Element,result) : Partition.Tree<&1,Move.Holding<Element>,Unit>}
  {==}

# ---- What a task holds ----

def onto_onto(~Element: Data,+stack: Held.Stack<Element>,+middle: Held.Stack<Element>,+after: Held.Stack<Element>) ->
  {Held.onto(~Element,Held.onto(~Element,stack,middle),after) == Held.onto(~Element,middle,Held.appended(~Element,stack,after)) : Held.Stack<Element>}:
  match stack:
    case Held.Bottom{}: {==}
    case Held.Held{+buffer,+first,+count,+below}: onto_onto(~Element,below,Held.Held{buffer,first,count,middle},after)

def onto_twice(~Element: Data,+stack: Held.Stack<Element>) ->
  {Held.onto(~Element,Held.onto(~Element,stack,Held.Bottom{}),Held.Bottom{}) == stack : Held.Stack<Element>}:
  Equal.trans(Held.Stack<Element>,Held.onto(~Element,Held.onto(~Element,stack,Held.Bottom{}),Held.Bottom{}),Held.appended(~Element,stack,Held.Bottom{}),stack,
    onto_onto(~Element,stack,Held.Bottom{},Held.Bottom{}),Held.appended_bottom(~Element,stack))

def boundary_spans(~Element: Data,+count: Nat,+stack: Held.Stack<Element>,+low: Nat) ->
  {Held.boundary(~Element,count,stack,low) == span_boundary(count,Held.spans(~Element,stack),low) : Nat}:
  match count stack:
    case 1n+ +rest Held.Held{+buffer,+first,+pieces,+below}: boundary_spans(~Element,rest,below,Nat.add(low,U32.to_nat(pieces)))
    case 0n stack: {==}
    case 1n+rest Held.Bottom{}: {==}

def boundary_above(~Element: Data,+count: Nat,+stack: Held.Stack<Element>,+low: Nat) -> {Nat.is_le(low,Held.boundary(~Element,count,stack,low)) == True{} : Bool}:
  match count stack:
    case 1n+ +rest Held.Held{+buffer,+first,+pieces,+below}:
      Order.transitive(low,Nat.add(low,U32.to_nat(pieces)),Held.boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces))),Order.le_add(low,U32.to_nat(pieces)),
        boundary_above(~Element,rest,below,Nat.add(low,U32.to_nat(pieces))))
    case 0n stack: Order.reflexive(low)
    case 1n+rest Held.Bottom{}: Order.reflexive(low)

def boundary_monotone(~Element: Data,+lower: Nat,+upper: Nat,+stack: Held.Stack<Element>,+low: Nat,+ordered: {Nat.is_le(lower,upper) == True{} : Bool}) ->
  {Nat.is_le(Held.boundary(~Element,lower,stack,low),Held.boundary(~Element,upper,stack,low)) == True{} : Bool}:
  %Order.restore_shift(lower,upper,ordered) :
    {Nat.is_le(Held.boundary(~Element,lower,stack,low),Held.boundary(~Element,_,stack,low)) == True{} : Bool}
  %Held.boundary_twice(~Element,lower,stack,Nat.sub(upper,lower),low) :
    {Nat.is_le(Held.boundary(~Element,lower,stack,low),_) == True{} : Bool}
  boundary_above(~Element,Nat.sub(upper,lower),Held.dropped(~Element,lower,stack),Held.boundary(~Element,lower,stack,low))

# A task holds every cell of its reach [start, finish): its pieces cover
# [start, high) with high at or after finish, with the values of the tensor.
type TaskWindow<-Element: Data,-Source: Data,-value: Source -> U32 -> Element,-source: Source,-holding: Held.Holding<Element>,-start: U32,-finish: U32,-total: Nat> is Data:
  TaskWindow{high: Nat,reaches: {Nat.is_le(U32.to_nat(finish),high) == True{} : Bool},inside: {Nat.is_le(high,total) == True{} : Bool},tiled: Held.Tiled(~Element,holding,U32.to_nat(start),high),
    values: Held.HoldingValues(~Element,~Source,~value,source,holding)}

# The window of a halo before pieces that cover the cells after it.
def window_from(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,+source: Source,+empty: Bool,+halo: Buffer.Model<Element>,+start: U32,+count: U32,
  +own: Held.Stack<Element>,+finish: U32,+high: Nat,+total: Nat,+inside: {Nat.is_le(high,total) == True{} : Bool},+zero: {U32.is_zero(count) == empty : Bool},
  +tiles: Held.Tiles(~Element,own,Nat.add(U32.to_nat(start),U32.to_nat(count)),high),+reaches: {Nat.is_le(U32.to_nat(finish),high) == True{} : Bool},
  +sized: {Buffer.length(~Element,halo) == U32.to_nat(count) : Nat},+ready: Buffer.ContentReady<Element,Buffer.content(~Element,halo)>,
  +piece: Content.Leaf(~Element,~Source,~value,source,start,count,Buffer.tree(~Element,halo),Buffer.length(~Element,halo)),
  +own_values: Held.StackValues(~Element,~Source,~value,source,own)) ->
  TaskWindow<Element,Source,value,source,window_holding(~Element,empty,halo,start,count,own),start,finish,total>:
  match empty:
    case True{}:
      +nothing = Arithmetic.word_zero(count,zero)
      TaskWindow{high,reaches,inside,
        Held.Tiling{U32.to_nat(start),{==},
          Held.tiles_moved(~Element,own,Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(start),high,
            Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(count)),Nat.add(U32.to_nat(start),0n),U32.to_nat(start),
              Equal.cong(U32,Nat,word => Nat.add(U32.to_nat(start),U32.to_nat(word)),count,0,nothing),Addition.zero_right(U32.to_nat(start))),
            tiles)},
        Held.Both{Unit{},own_values}}
    case False{}:
      TaskWindow{high,reaches,inside,Held.Tiling{U32.to_nat(start),{==},Held.Both{Held.Starts{{==},sized,ready},tiles}},Held.Both{Unit{},Held.Both{piece,own_values}}}

def sub_below(+value: Nat,+bound: Nat,+order: {Nat.is_le(value,bound) == True{} : Bool}) -> {Nat.sub(value,bound) == 0n : Nat}:
  match value bound:
    case 0n bound: {==}
    case 1n+rest 0n: Empty.absurd({Nat.sub(1n+rest,0n) == 0n : Nat},WordEncoding.false_true(order))
    case 1n+ +rest 1n+ +below: sub_below(rest,below,order)

def below_maximum(+left: U32,+right: U32,+choice: Bool,+test: {U32.is_lt(left,right) == choice : Bool},
  +less: {Nat.is_lt(U32.to_nat(left),U32.to_nat(Bool.pick(U32,choice,right,left))) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}:
  match choice:
    case True{}: less
    case False{}: Empty.absurd({Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool},
      Held.lt_le_absurd(U32.to_nat(left),U32.to_nat(left),less,Order.reflexive(U32.to_nat(left))))

def maximum_above(+left: U32,+right: U32,+less: {Nat.is_lt(U32.to_nat(left),U32.to_nat(U32.max(left,right))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}:
  below_maximum(left,right,U32.is_lt(left,right),{==},{%Equal.sym(U32,Bool.pick(U32,U32.is_lt(left,right),right,left),U32.max(left,right),Extrema.pick_maximum(left,right)) :
      {Nat.is_lt(U32.to_nat(left),U32.to_nat(_)) == True{} : Bool}
    less : {Nat.is_lt(U32.to_nat(left),U32.to_nat(Bool.pick(U32,U32.is_lt(left,right),right,left))) == True{} : Bool}})

# The halo ends where its cells end: start + count is its end.
def halo_exact(+start: U32,+reach: U32,+finish: U32,+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+total: Nat,
  +small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},+finish_inside: {Nat.is_le(U32.to_nat(finish),total) == True{} : Bool},+between: Between<start,reach,finish>) ->
  {Nat.add(U32.to_nat(start),U32.to_nat((reach - start : U32))) == U32.to_nat(reach) : Nat}:
  Between{+above,+below} = between
  Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat((reach - start : U32))),Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(reach),U32.to_nat(start))),U32.to_nat(reach),
    Equal.cong(Nat,Nat,count => Nat.add(U32.to_nat(start),count),U32.to_nat((reach - start : U32)),Nat.sub(U32.to_nat(reach),U32.to_nat(start)),
      Arithmetic.subtract(reach,start,limit,Coordinates.native_le_transitive(limit,1073741824,2147483648,limit_ok,{==}),above,
        Order.transitive(Nat.sub(U32.to_nat(reach),U32.to_nat(start)),U32.to_nat(reach),U32.to_nat(limit),Order.sub_le(U32.to_nat(reach),U32.to_nat(start)),
          Order.transitive(U32.to_nat(reach),total,U32.to_nat(limit),Order.transitive(U32.to_nat(reach),U32.to_nat(finish),total,below,finish_inside),small)))),
    Order.restore_shift(U32.to_nat(start),U32.to_nat(reach),above))

# The window of a task whose pieces start at low: the halo up to low, or up to
# the end of its reach when it owns no piece.
def task_window(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,
  +source: Source,+order: Held.Stack<Element>,+halo: Buffer.Model<Element>,+start: U32,+finish: U32,+middle: U32,+choice: Bool,
  +test: {U32.is_lt(Halo.piece_from(grid,middle),U32.max(Halo.piece_from(grid,middle),Halo.piece_from(grid,finish))) == choice : Bool},
  +exact: {Nat.add(U32.to_nat(start),U32.to_nat((Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish) - start : U32))) ==
    U32.to_nat(Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish)) : Nat},
  +tiles: Held.Tiles(~Element,order,0n,total),+values: Held.StackValues(~Element,~Source,~value,source,order),
  +ordered: {Held.spans(~Element,order) == layout : Held.Spans},+finish_inside: {Nat.is_le(U32.to_nat(finish),total) == True{} : Bool},
  +low_high: {Nat.is_le(U32.to_nat(Halo.piece_from(grid,middle)),U32.to_nat(U32.max(Halo.piece_from(grid,middle),Halo.piece_from(grid,finish)))) == True{} : Bool},
  +sized: {Buffer.length(~Element,halo) == U32.to_nat((Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish) - start : U32)) : Nat},
  +ready: Buffer.ContentReady<Element,Buffer.content(~Element,halo)>,
  +piece: Content.Leaf(~Element,~Source,~value,source,start,(Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish) - start : U32),
    Buffer.tree(~Element,halo),Buffer.length(~Element,halo))) ->
  TaskWindow<Element,Source,value,source,
    window_holding(~Element,U32.is_zero((Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish) - start : U32)),halo,start,
      (Bool.pick(U32,choice,U32.min(Grid.start(grid,Halo.piece_from(grid,middle)),finish),finish) - start : U32),
      Held.taken(~Element,Nat.sub(U32.to_nat(U32.max(Halo.piece_from(grid,middle),Halo.piece_from(grid,finish))),U32.to_nat(Halo.piece_from(grid,middle))),
        Held.dropped(~Element,U32.to_nat(Halo.piece_from(grid,middle)),order))),start,finish,total>:
  match choice:
    case True{}:
      +low = Halo.piece_from(grid,middle)
      +high = U32.max(low,Halo.piece_from(grid,finish))
      +keep = Nat.sub(U32.to_nat(high),U32.to_nat(low))
      +rest = Held.dropped(~Element,U32.to_nat(low),order)
      +count = (U32.min(Grid.start(grid,low),finish) - start : U32)
      +low_at = Held.boundary(~Element,U32.to_nat(low),order,0n)
      +after_low = Held.second_of(Held.Tiles(~Element,Held.taken(~Element,U32.to_nat(low),order),0n,Held.boundary(~Element,U32.to_nat(low),order,0n)),Held.Tiles(~Element,Held.dropped(~Element,U32.to_nat(low),order),Held.boundary(~Element,U32.to_nat(low),order,0n),total),Held.split_tiles(~Element,U32.to_nat(low),order,0n,total,tiles))
      +own_tiles = Held.first_of(Held.Tiles(~Element,Held.taken(~Element,keep,rest),Held.boundary(~Element,U32.to_nat(low),order,0n),Held.boundary(~Element,keep,rest,Held.boundary(~Element,U32.to_nat(low),order,0n))),Held.Tiles(~Element,Held.dropped(~Element,keep,rest),Held.boundary(~Element,keep,rest,Held.boundary(~Element,U32.to_nat(low),order,0n)),total),Held.split_tiles(~Element,keep,rest,low_at,total,after_low))
      +values_rest = Held.second_of(Held.StackValues(~Element,~Source,~value,source,Held.taken(~Element,U32.to_nat(low),order)),Held.StackValues(~Element,~Source,~value,source,Held.dropped(~Element,U32.to_nat(low),order)),Held.split_values(~Element,~Source,~value,source,U32.to_nat(low),order,values))
      +own_values = Held.first_of(Held.StackValues(~Element,~Source,~value,source,Held.taken(~Element,keep,rest)),Held.StackValues(~Element,~Source,~value,source,Held.dropped(~Element,keep,rest)),Held.split_values(~Element,~Source,~value,source,keep,rest,values_rest))
      +high_bounded = Bounds.as_nat_le(high,Grid.pieces_of(grid),
        Extrema.maximum_upper(low,Halo.piece_from(grid,finish),Grid.pieces_of(grid),Laws.from_bounded(grid,middle),Laws.from_bounded(grid,finish)))
      +low_inside = Bounds.nat_trans(U32.to_nat(low),U32.to_nat(high),U32.to_nat(Grid.pieces_of(grid)),Bounds.as_nat_lt(low,high,test),high_bounded)
      +low_before = maximum_above(low,Halo.piece_from(grid,finish),Bounds.as_nat_lt(low,high,test))
      +start_low = Bounds.strict_le(U32.to_nat(Grid.start(grid,low)),U32.to_nat(finish),before_of(limits,grid,layout,total,laid,finish,finish_inside,low,low_before))
      +halo_end = Equal.trans(Nat,U32.to_nat(U32.min(Grid.start(grid,low),finish)),Nat.min(U32.to_nat(Grid.start(grid,low)),U32.to_nat(finish)),U32.to_nat(Grid.start(grid,low)),
        Extrema.minimum(Grid.start(grid,low),finish),NatExtrema.minimum_when_ordered(U32.to_nat(Grid.start(grid,low)),U32.to_nat(finish),start_low))
      +low_boundary = Equal.trans(Nat,low_at,span_boundary(U32.to_nat(low),Held.spans(~Element,order),0n),U32.to_nat(Grid.start(grid,low)),
        boundary_spans(~Element,U32.to_nat(low),order,0n),
        Equal.trans(Nat,span_boundary(U32.to_nat(low),Held.spans(~Element,order),0n),span_boundary(U32.to_nat(low),layout,0n),U32.to_nat(Grid.start(grid,low)),
          Equal.cong(Held.Spans,Nat,spans => span_boundary(U32.to_nat(low),spans,0n),Held.spans(~Element,order),layout,ordered),starts_of(grid,layout,laid_along(limits,grid,layout,total,laid),low,low_inside)))
      +halo_boundary = Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(Grid.start(grid,low)),low_at,
        Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(U32.min(Grid.start(grid,low),finish)),U32.to_nat(Grid.start(grid,low)),exact,halo_end),
        Equal.sym(Nat,low_at,U32.to_nat(Grid.start(grid,low)),low_boundary))
      +own_end = Held.boundary(~Element,keep,rest,low_at)
      +own_inside = Held.tiles_ordered(~Element,Held.dropped(~Element,keep,rest),own_end,total,Held.second_of(Held.Tiles(~Element,Held.taken(~Element,keep,rest),Held.boundary(~Element,U32.to_nat(low),order,0n),Held.boundary(~Element,keep,rest,Held.boundary(~Element,U32.to_nat(low),order,0n))),Held.Tiles(~Element,Held.dropped(~Element,keep,rest),Held.boundary(~Element,keep,rest,Held.boundary(~Element,U32.to_nat(low),order,0n)),total),Held.split_tiles(~Element,keep,rest,low_at,total,after_low)))
      +own_end_high = Equal.trans(Nat,own_end,Held.boundary(~Element,Nat.add(U32.to_nat(low),keep),order,0n),Held.boundary(~Element,U32.to_nat(high),order,0n),
        Held.boundary_twice(~Element,U32.to_nat(low),order,keep,0n),
        Equal.cong(Nat,Nat,count => Held.boundary(~Element,count,order,0n),Nat.add(U32.to_nat(low),keep),U32.to_nat(high),
          Order.restore_shift(U32.to_nat(low),U32.to_nat(high),low_high)))
      +from_finish = Bounds.as_nat_le(Halo.piece_from(grid,finish),high,Extrema.maximum_right(low,Halo.piece_from(grid,finish)))
      +finish_boundary = Held.nat_le_right_moved(U32.to_nat(finish),Held.boundary(~Element,U32.to_nat(Halo.piece_from(grid,finish)),order,0n),
        span_boundary(U32.to_nat(Halo.piece_from(grid,finish)),layout,0n),
        Equal.trans(Nat,Held.boundary(~Element,U32.to_nat(Halo.piece_from(grid,finish)),order,0n),span_boundary(U32.to_nat(Halo.piece_from(grid,finish)),Held.spans(~Element,order),0n),
          span_boundary(U32.to_nat(Halo.piece_from(grid,finish)),layout,0n),boundary_spans(~Element,U32.to_nat(Halo.piece_from(grid,finish)),order,0n),
          Equal.cong(Held.Spans,Nat,spans => span_boundary(U32.to_nat(Halo.piece_from(grid,finish)),spans,0n),Held.spans(~Element,order),layout,ordered)),
        reached_of(limits,grid,layout,total,laid,finish,finish_inside))
      +reaches = Held.nat_le_right_moved(U32.to_nat(finish),own_end,Held.boundary(~Element,U32.to_nat(high),order,0n),own_end_high,
        Order.transitive(U32.to_nat(finish),Held.boundary(~Element,U32.to_nat(Halo.piece_from(grid,finish)),order,0n),Held.boundary(~Element,U32.to_nat(high),order,0n),
          finish_boundary,boundary_monotone(~Element,U32.to_nat(Halo.piece_from(grid,finish)),U32.to_nat(high),order,0n,from_finish)))
      window_from(~Element,~Source,~value,source,U32.is_zero(count),halo,start,count,Held.taken(~Element,keep,rest),finish,own_end,total,own_inside,{==},
        Held.tiles_moved(~Element,Held.taken(~Element,keep,rest),low_at,Nat.add(U32.to_nat(start),U32.to_nat(count)),own_end,
          Equal.sym(Nat,Nat.add(U32.to_nat(start),U32.to_nat(count)),low_at,halo_boundary),own_tiles),
        reaches,sized,ready,piece,own_values)
    case False{}:
      +low = Halo.piece_from(grid,middle)
      +high = U32.max(low,Halo.piece_from(grid,finish))
      +keep = Nat.sub(U32.to_nat(high),U32.to_nat(low))
      +rest = Held.dropped(~Element,U32.to_nat(low),order)
      +count = (finish - start : U32)
      +not_less = Held.lt_false_le(U32.to_nat(low),U32.to_nat(high),Held.nat_lt_false(low,high,test))
      %Equal.sym(Nat,keep,0n,sub_below(U32.to_nat(high),U32.to_nat(low),not_less)) :
        TaskWindow<Element,Source,value,source,window_holding(~Element,U32.is_zero(count),halo,start,count,Held.taken(~Element,_,rest)),start,finish,total>
      window_from(~Element,~Source,~value,source,U32.is_zero(count),halo,start,count,Held.Bottom{},finish,Nat.add(U32.to_nat(start),U32.to_nat(count)),total,
        Held.nat_le_moved(Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(finish),total,exact,finish_inside),{==},{==},
        Held.nat_le_right_moved(U32.to_nat(finish),Nat.add(U32.to_nat(start),U32.to_nat(count)),U32.to_nat(finish),exact,Order.reflexive(U32.to_nat(finish))),
        sized,ready,piece,Unit{})

# Every task of tasks holds its reach.
def WindowsOk(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+source: Source,+task: Task,+total: Nat,tasks: Pieces.Ranges,
  windows: WindowModel<Element>) -> Data:
  match tasks windows:
    case Pieces.Range{+first,+count} WindowLeaf{+holding,+task_first,+task_count}:
      Held.Both<{task_first == first : U32},Held.Both<{task_count == count : U32},
        TaskWindow<Element,Source,value,source,holding,span_start(reach(task,first,count)),span_end(reach(task,first,count)),total>>>
    case Pieces.Split{+left,+right} WindowFork{+left_windows,+right_windows}:
      Held.Both<WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,left,left_windows),WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,right,right_windows)>
    case tasks windows: Empty

# After the tasks: their windows, the pieces not yet handed are those of the
# order from the next index on, and that index is before every later low.
type Handling<-Element: Data,-Source: Data,-value: Source -> U32 -> Element,-source: Source,-Task: Data,-reach: Halo.Reach(~Task),-task: Task,-grid: Grid.Grid,-total: Nat,
  -tasks: Pieces.Ranges,-reached: U32,-order: Held.Stack<Element>,-handled: Handled<Element>> is Data:
  Handling{windows: WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,tasks,handled_windows(~Element,handled)),
    stack: {handled_stack(~Element,handled) == Held.dropped(~Element,U32.to_nat(handled_next(~Element,handled)),order) : Held.Stack<Element>},
    next: {Nat.is_le(U32.to_nat(handled_next(~Element,handled)),U32.to_nat(Halo.piece_from(grid,reach_after(~Task,~reach,task,tasks,reached)))) == True{} : Bool}}

def bounds_fits(+start: U32,+count: U32,+total: Nat,+bounds: HaloBounds<start,count,total>) -> {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool}:
  HaloBounds{fits,allocatable} = bounds
  fits

def bounds_allocatable(+start: U32,+count: U32,+total: Nat,+bounds: HaloBounds<start,count,total>) -> {U32.is_le(count,1073741824) == True{} : Bool}:
  HaloBounds{fits,allocatable} = bounds
  allocatable

# One task: its window holds its reach and the walk goes on after its pieces.
def leaf_step(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,
  +source: Source,+task: Task,+fill: Element,+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},
  +pieces_ok: {U32.is_le(Grid.pieces_of(grid),2147483648) == True{} : Bool},+task_first: U32,+task_count: U32,+holding: Held.Holding<Element>,+reached: U32,+next: U32,
  +facts: HeldFacts<Element,Source,value,source,holding,total,layout>,
  +next_low: {Nat.is_le(U32.to_nat(next),U32.to_nat(Halo.piece_from(grid,reached))) == True{} : Bool},
  +reached_inside: {Nat.is_le(U32.to_nat(reached),total) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,task_first,task_count))),
    Nat.sub(U32.to_nat(span_end(reach(task,task_first,task_count))),U32.to_nat(span_start(reach(task,task_first,task_count))))),total) == True{} : Bool},
  +ordered_span: {Nat.is_le(U32.to_nat(span_start(reach(task,task_first,task_count))),U32.to_nat(span_end(reach(task,task_first,task_count)))) == True{} : Bool}) ->
  Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),handled_model(~Element,walked_halos(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,Pieces.Range{task_first,task_count},holding,reached)),Held.dropped(~Element,U32.to_nat(next),Held.order(~Element,holding)),next)>:
  HeldFacts{+tiled,+values,+ordered} = facts
  +span = reach(task,task_first,task_count)
  +start = span_start(span)
  +finish = span_end(span)
  +middle = U32.max(start,reached)
  +low = Halo.piece_from(grid,middle)
  +high = U32.max(low,Halo.piece_from(grid,finish))
  +count = halo_count(grid,span,reached)
  +halo = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,span,holding,reached)))
  +order = Held.order(~Element,holding)
  +stack = Held.dropped(~Element,U32.to_nat(next),order)
  +finish_inside = Held.nat_le_moved(U32.to_nat(finish),Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),total,
    Equal.sym(Nat,Nat.add(U32.to_nat(start),Nat.sub(U32.to_nat(finish),U32.to_nat(start))),U32.to_nat(finish),Order.restore_shift(U32.to_nat(start),U32.to_nat(finish),ordered_span)),
    inside)
  +middle_inside = maximum_inside(start,reached,total,Order.transitive(U32.to_nat(start),U32.to_nat(finish),total,ordered_span,finish_inside),reached_inside)
  +after_reached = Bounds.as_nat_le(reached,middle,Extrema.maximum_right(start,reached))
  +next_before = Order.transitive(U32.to_nat(next),U32.to_nat(Halo.piece_from(grid,reached)),U32.to_nat(low),next_low,monotone_of(limits,grid,layout,total,laid,reached,middle,middle_inside,after_reached))
  +low_high = Bounds.as_nat_le(low,high,Extrema.maximum_left(low,Halo.piece_from(grid,finish)))
  +high_bounded = Extrema.maximum_upper(low,Halo.piece_from(grid,finish),Grid.pieces_of(grid),Laws.from_bounded(grid,middle),Laws.from_bounded(grid,finish))
  +low_bounded = Bounds.as_nat_le(low,Grid.pieces_of(grid),Laws.from_bounded(grid,middle))
  +skip_exact = Arithmetic.subtract(low,next,Grid.pieces_of(grid),pieces_ok,next_before,
    Order.transitive(Nat.sub(U32.to_nat(low),U32.to_nat(next)),U32.to_nat(low),U32.to_nat(Grid.pieces_of(grid)),Order.sub_le(U32.to_nat(low),U32.to_nat(next)),low_bounded))
  +keep_exact = Arithmetic.subtract(high,low,Grid.pieces_of(grid),pieces_ok,low_high,
    Order.transitive(Nat.sub(U32.to_nat(high),U32.to_nat(low)),U32.to_nat(high),U32.to_nat(Grid.pieces_of(grid)),Order.sub_le(U32.to_nat(high),U32.to_nat(low)),
      Bounds.as_nat_le(high,Grid.pieces_of(grid),high_bounded)))
  +keep = Nat.sub(U32.to_nat(high),U32.to_nat(low))
  +rest_exact = Equal.trans(Held.Stack<Element>,Held.dropped(~Element,U32.to_nat((low - next : U32)),stack),
    Held.dropped(~Element,Nat.add(U32.to_nat(next),U32.to_nat((low - next : U32))),order),Held.dropped(~Element,U32.to_nat(low),order),
    Held.dropped_twice(~Element,U32.to_nat(next),order,U32.to_nat((low - next : U32))),
    Equal.cong(Nat,Held.Stack<Element>,skip => Held.dropped(~Element,skip,order),Nat.add(U32.to_nat(next),U32.to_nat((low - next : U32))),U32.to_nat(low),
      Equal.trans(Nat,Nat.add(U32.to_nat(next),U32.to_nat((low - next : U32))),Nat.add(U32.to_nat(next),Nat.sub(U32.to_nat(low),U32.to_nat(next))),U32.to_nat(low),
        Equal.cong(Nat,Nat,skip => Nat.add(U32.to_nat(next),skip),U32.to_nat((low - next : U32)),Nat.sub(U32.to_nat(low),U32.to_nat(next)),skip_exact),
        Order.restore_shift(U32.to_nat(next),U32.to_nat(low),next_before))))
  +rest = Held.dropped(~Element,U32.to_nat(low),order)
  +bounds = halo_bounds(limits,grid,layout,total,laid,span,reached,limit,limit_ok,small,inside,ordered_span,reached_inside)
  +reached_next = U32.max(reached,finish)
  +next_inside = maximum_inside(reached,finish,total,reached_inside,finish_inside)
  +middle_next = Bounds.as_nat_le(middle,reached_next,Extrema.maximum_upper(start,reached,reached_next,
    Coordinates.native_le_transitive(start,finish,reached_next,Held.u32_le(start,finish,ordered_span),Extrema.maximum_right(reached,finish)),Extrema.maximum_left(reached,finish)))
  +finish_next = Bounds.as_nat_le(finish,reached_next,Extrema.maximum_right(reached,finish))
  %Equal.sym(U32,U32.max(low,next),low,Laws.maximum_kept(low,next,next_before)) :
    Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),handled_at(~Element,halo,start,count,task_first,task_count,stack,next,_,U32.max(high,_))>
  %Equal.sym(U32,U32.max(high,low),high,Laws.maximum_kept(high,low,low_high)) :
    Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),handled_at(~Element,halo,start,count,task_first,task_count,stack,next,low,_)>
  %Equal.sym(Held.Stack<Element>,Held.onto(~Element,Held.onto(~Element,Held.taken(~Element,U32.to_nat((high - low : U32)),Held.dropped(~Element,U32.to_nat((low - next : U32)),stack)),Held.Bottom{}),Held.Bottom{}),
    Held.taken(~Element,U32.to_nat((high - low : U32)),Held.dropped(~Element,U32.to_nat((low - next : U32)),stack)),
    onto_twice(~Element,Held.taken(~Element,U32.to_nat((high - low : U32)),Held.dropped(~Element,U32.to_nat((low - next : U32)),stack)))) :
    Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),Handled{WindowLeaf{window_holding(~Element,U32.is_zero(count),halo,start,count,_),task_first,task_count},Held.dropped(~Element,U32.to_nat((high - low : U32)),Held.dropped(~Element,U32.to_nat((low - next : U32)),stack)),high}>
  %Equal.sym(Held.Stack<Element>,Held.dropped(~Element,U32.to_nat((low - next : U32)),stack),rest,rest_exact) :
    Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),Handled{WindowLeaf{window_holding(~Element,U32.is_zero(count),halo,start,count,Held.taken(~Element,U32.to_nat((high - low : U32)),_)),task_first,task_count},Held.dropped(~Element,U32.to_nat((high - low : U32)),_),high}>
  %Equal.sym(Nat,U32.to_nat((high - low : U32)),keep,keep_exact) :
    Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{task_first,task_count},reached,Held.order(~Element,holding),Handled{WindowLeaf{window_holding(~Element,U32.is_zero(count),halo,start,count,Held.taken(~Element,_,rest)),task_first,task_count},Held.dropped(~Element,_,rest),high}>
  Handling{Held.Both{{==},Held.Both{{==},
    task_window(~Element,~Source,~value,limits,grid,layout,total,laid,source,order,halo,start,finish,middle,U32.is_lt(low,high),{==},
      halo_exact(start,halo_end(grid,span,reached),finish,limit,limit_ok,total,small,finish_inside,
        end_between(limits,grid,layout,total,laid,start,reached,finish,U32.is_lt(low,high),{==},ordered_span,middle_inside)),
      Held.order_tiles(~Element,holding,0n,total,tiled),Held.order_values(~Element,~Source,~value,source,holding,values),ordered,finish_inside,low_high,
      Equal.trans(Nat,Buffer.length(~Element,halo),Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,span,holding,reached)))),
        U32.to_nat(count),Held.copy_length(~Element,holding,start,count,fill),Held.copied_length(~Element,holding,start,count,fill)),
      {%Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,halo),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,span,holding,reached))),Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,halo_copy(~Element,grid,fill,span,holding,reached)))) :
        Buffer.ContentReady<Element,_>
        Held.copied_ready(~Element,holding,start,count,fill,bounds_allocatable(start,count,total,bounds)) : Buffer.ContentReady<Element,Buffer.content(~Element,halo)>},
      Held.copied_leaf(~Element,~Source,~value,source,holding,start,count,fill,0n,total,limit,Coordinates.native_le_transitive(limit,1073741824,2147483648,limit_ok,{==}),small,tiled,values,
        Order.zero_le(U32.to_nat(start)),bounds_fits(start,count,total,bounds),bounds_allocatable(start,count,total,bounds)))}},
    Equal.trans(Held.Stack<Element>,Held.dropped(~Element,keep,rest),Held.dropped(~Element,Nat.add(U32.to_nat(low),keep),order),Held.dropped(~Element,U32.to_nat(high),order),
      Held.dropped_twice(~Element,U32.to_nat(low),order,keep),
      Equal.cong(Nat,Held.Stack<Element>,taken => Held.dropped(~Element,taken,order),Nat.add(U32.to_nat(low),keep),U32.to_nat(high),Order.restore_shift(U32.to_nat(low),U32.to_nat(high),low_high))),
    Bounds.as_nat_le(high,Halo.piece_from(grid,reached_next),Extrema.maximum_upper(low,Halo.piece_from(grid,finish),Halo.piece_from(grid,reached_next),
      Held.u32_le(low,Halo.piece_from(grid,reached_next),monotone_of(limits,grid,layout,total,laid,middle,reached_next,next_inside,middle_next)),
      Held.u32_le(Halo.piece_from(grid,finish),Halo.piece_from(grid,reached_next),monotone_of(limits,grid,layout,total,laid,finish,reached_next,next_inside,finish_next))))}

def handling_stack(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+total: Nat,+source: Source,
  +tasks: Pieces.Ranges,+reached: U32,+order: Held.Stack<Element>,+handled: Handled<Element>,+handling: Handling<Element,Source,value,source,Task,reach,task,grid,total,tasks,reached,order,handled>) ->
  {handled_stack(~Element,handled) == Held.dropped(~Element,U32.to_nat(handled_next(~Element,handled)),order) : Held.Stack<Element>}:
  Handling{windows,stack,next} = handling
  stack

def handling_next(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+total: Nat,+source: Source,
  +tasks: Pieces.Ranges,+reached: U32,+order: Held.Stack<Element>,+handled: Handled<Element>,+handling: Handling<Element,Source,value,source,Task,reach,task,grid,total,tasks,reached,order,handled>) ->
  {Nat.is_le(U32.to_nat(handled_next(~Element,handled)),U32.to_nat(Halo.piece_from(grid,reach_after(~Task,~reach,task,tasks,reached)))) == True{} : Bool}:
  Handling{windows,stack,next} = handling
  next

def handling_windows(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+total: Nat,+source: Source,
  +tasks: Pieces.Ranges,+reached: U32,+order: Held.Stack<Element>,+handled: Handled<Element>,+handling: Handling<Element,Source,value,source,Task,reach,task,grid,total,tasks,reached,order,handled>) ->
  WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,tasks,handled_windows(~Element,handled)):
  Handling{windows,stack,next} = handling
  windows

def fork_handling(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+task: Task,+grid: Grid.Grid,+total: Nat,+source: Source,
  +left: Pieces.Ranges,+right: Pieces.Ranges,+reached: U32,+order: Held.Stack<Element>,+left_handled: Handled<Element>,+right_handled: Handled<Element>,
  +left_handling: Handling<Element,Source,value,source,Task,reach,task,grid,total,left,reached,order,left_handled>,
  +right_handling: Handling<Element,Source,value,source,Task,reach,task,grid,total,right,reach_after(~Task,~reach,task,left,reached),order,right_handled>) ->
  Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Split{left,right},reached,order,handled_join(~Element,handled_windows(~Element,left_handled),right_handled)>:
  match right_handled:
    case Handled{+windows,+stack,+next}:
      Handling{+left_windows,left_stack,left_next} = left_handling
      Handling{+right_windows,+right_stack,+right_next} = right_handling
      Handling{Held.Both{left_windows,right_windows},right_stack,right_next}

# Every task of tasks holds its reach, from a walk state where the pieces not
# yet handed are the order from next on.
def tasks_step(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,+task: Task,+source: Source,+fill: Element,+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},+pieces_ok: {U32.is_le(Grid.pieces_of(grid),2147483648) == True{} : Bool},
  +tasks: Pieces.Ranges,+reaches: Reaches(~Task,~reach,task,tasks,total),+holding: Held.Holding<Element>,+reached: U32,+stack: Held.Stack<Element>,+next: U32,+facts: HeldFacts<Element,Source,value,source,holding,total,layout>,
  +next_low: {Nat.is_le(U32.to_nat(next),U32.to_nat(Halo.piece_from(grid,reached))) == True{} : Bool},
  +reached_inside: {Nat.is_le(U32.to_nat(reached),total) == True{} : Bool},
  +at: {stack == Held.dropped(~Element,U32.to_nat(next),Held.order(~Element,holding)) : Held.Stack<Element>}) ->
  Handling<Element,Source,value,source,Task,reach,task,grid,total,tasks,reached,Held.order(~Element,holding),handled_model(~Element,walked_halos(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,reached)),stack,next)>:
  match tasks:
    case Pieces.Range{+first,+count}:
      %Equal.sym(Held.Stack<Element>,stack,Held.dropped(~Element,U32.to_nat(next),Held.order(~Element,holding)),at) :
        Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Range{first,count},reached,Held.order(~Element,holding),handled_model(~Element,walked_halos(~Element,walk_model(~Element,~Task,~reach,task,grid,fill,Pieces.Range{first,count},holding,reached)),_,next)>
      leaf_step(~Element,~Source,~value,~Task,~reach,limits,grid,layout,total,laid,source,task,fill,limit,limit_ok,small,pieces_ok,first,count,holding,reached,next,facts,next_low,reached_inside,Held.first_of({Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,first,count))),Nat.sub(U32.to_nat(span_end(reach(task,first,count))),U32.to_nat(span_start(reach(task,first,count))))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(span_start(reach(task,first,count))),U32.to_nat(span_end(reach(task,first,count)))) == True{} : Bool},reaches),Held.second_of({Nat.is_le(Nat.add(U32.to_nat(span_start(reach(task,first,count))),Nat.sub(U32.to_nat(span_end(reach(task,first,count))),U32.to_nat(span_start(reach(task,first,count))))),total) == True{} : Bool},{Nat.is_le(U32.to_nat(span_start(reach(task,first,count))),U32.to_nat(span_end(reach(task,first,count)))) == True{} : Bool},reaches))
    case Pieces.Split{+left,+right}:
      +left_walk = walk_model(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      +left_reached = reach_after(~Task,~reach,task,left,reached)
      +right_walk = walk_model(~Element,~Task,~reach,task,grid,fill,right,walked_holding(~Element,left_walk),left_reached)
      +left_reaches = Held.first_of(Reaches(~Task,~reach,task,left,total),Reaches(~Task,~reach,task,right,total),reaches)
      +right_reaches = Held.second_of(Reaches(~Task,~reach,task,left,total),Reaches(~Task,~reach,task,right,total),reaches)
      +left_handled = handled_model(~Element,walked_halos(~Element,left_walk),stack,next)
      +right_handled = handled_model(~Element,walked_halos(~Element,right_walk),handled_stack(~Element,left_handled),handled_next(~Element,left_handled))
      +left_handling = tasks_step(~Element,~Source,~value,~Task,~reach,limits,grid,layout,total,laid,task,source,fill,limit,limit_ok,small,pieces_ok,left,left_reaches,holding,reached,stack,next,facts,next_low,reached_inside,at)
      +left_order = walk_order(~Element,~Task,~reach,task,grid,fill,left,holding,reached)
      +right_handling = tasks_step(~Element,~Source,~value,~Task,~reach,limits,grid,layout,total,laid,task,source,fill,limit,limit_ok,small,pieces_ok,right,right_reaches,walked_holding(~Element,left_walk),left_reached,handled_stack(~Element,left_handled),handled_next(~Element,left_handled),
        walk_facts(~Element,~Source,~value,~Task,~reach,source,task,grid,fill,left,holding,reached,total,layout,facts),
        handling_next(~Element,~Source,~value,~Task,~reach,task,grid,total,source,left,reached,Held.order(~Element,holding),left_handled,left_handling),
        reach_after_inside(~Task,~reach,task,total,left,left_reaches,reached,reached_inside),
        Equal.trans(Held.Stack<Element>,handled_stack(~Element,left_handled),Held.dropped(~Element,U32.to_nat(handled_next(~Element,left_handled)),Held.order(~Element,holding)),
          Held.dropped(~Element,U32.to_nat(handled_next(~Element,left_handled)),Held.order(~Element,walked_holding(~Element,left_walk))),
          handling_stack(~Element,~Source,~value,~Task,~reach,task,grid,total,source,left,reached,Held.order(~Element,holding),left_handled,left_handling),
          Equal.cong(Held.Stack<Element>,Held.Stack<Element>,pieces => Held.dropped(~Element,U32.to_nat(handled_next(~Element,left_handled)),pieces),Held.order(~Element,holding),
            Held.order(~Element,walked_holding(~Element,left_walk)),Equal.sym(Held.Stack<Element>,Held.order(~Element,walked_holding(~Element,left_walk)),Held.order(~Element,holding),left_order))))
      %Equal.sym(HaloModel<Element>,walked_halos(~Element,walked_join(~Element,walked_halos(~Element,left_walk),right_walk)),HaloFork{walked_halos(~Element,left_walk),walked_halos(~Element,right_walk)},
        join_halos(~Element,walked_halos(~Element,left_walk),right_walk)) :
        Handling<Element,Source,value,source,Task,reach,task,grid,total,Pieces.Split{left,right},reached,Held.order(~Element,holding),handled_model(~Element,_,stack,next)>
      fork_handling(~Element,~Source,~value,~Task,~reach,task,grid,total,source,left,right,reached,Held.order(~Element,holding),left_handled,right_handled,left_handling,
        {%left_order : Handling<Element,Source,value,source,Task,reach,task,grid,total,right,left_reached,_,right_handled>
          right_handling : Handling<Element,Source,value,source,Task,reach,task,grid,total,right,left_reached,Held.order(~Element,holding),right_handled>})

# The windows of Halo.windows: the pack of a model in which every task holds
# the cells of its reach with the values of the tensor.
type Windowed<-Element: Data,-Source: Data,-value: Source -> U32 -> Element,-source: Source,-Task: Data,-reach: Halo.Reach(~Task),-task: Task,-total: Nat,-tasks: Pieces.Ranges,
  -windows: Partition.Tree<&1,Move.Holding<Element>,Unit>> is Data:
  Windowed{model: WindowModel<Element>,packed: {windows == pack_windows(~Element,model) : Partition.Tree<&1,Move.Holding<Element>,Unit>},
    holds: WindowsOk(~Element,~Source,~value,~Task,~reach,source,task,total,tasks,model)}

# The windows of a source are the windows of a holder that projects to it.
def windows_projected(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Holder: Data,~project: Holder -> Source,~Task: Data,~reach: Halo.Reach(~Task),+holder: Holder,+task: Task,+total: Nat,tasks: Pieces.Ranges,windows: WindowModel<Element>,
  ok: WindowsOk(~Element,~Source,~value,~Task,~reach,project(holder),task,total,tasks,windows)) ->
  WindowsOk(~Element,~Holder,~Held.projected(~Element,~Source,~value,~Holder,~project),~Task,~reach,holder,task,total,tasks,windows):
  match tasks windows:
    case Pieces.Range{+first,+count} WindowLeaf{+holding,+task_first,+task_count}:
      Held.Both{+first_same,+rest} = ok
      Held.Both{+count_same,+window} = rest
      TaskWindow{+high,+reaches,+inside,+tiled,+values} = window
      Held.Both{first_same,Held.Both{count_same,TaskWindow{high,reaches,inside,tiled,Held.holding_projected(~Element,~Source,~value,~Holder,~project,holder,holding,values)}}}
    case Pieces.Split{+left,+right} WindowFork{+left_windows,+right_windows}:
      Held.Both{+left_ok,+right_ok} = ok
      Held.Both{windows_projected(~Element,~Source,~value,~Holder,~project,~Task,~reach,holder,task,total,left,left_windows,left_ok),windows_projected(~Element,~Source,~value,~Holder,~project,~Task,~reach,holder,task,total,right,right_windows,right_ok)}
    case Pieces.Range{+first,+count} WindowFork{+left_windows,+right_windows}: ok
    case Pieces.Split{+left,+right} WindowLeaf{+holding,+task_first,+task_count}: ok

def windowed_projected(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Holder: Data,~project: Holder -> Source,~Task: Data,~reach: Halo.Reach(~Task),+holder: Holder,+task: Task,+total: Nat,+tasks: Pieces.Ranges,-windows: Partition.Tree<&1,Move.Holding<Element>,Unit>,
  windowed: Windowed<Element,Source,value,project(holder),Task,reach,task,total,tasks,windows>) ->
  Windowed<Element,Holder,Held.projected(~Element,~Source,~value,~Holder,~project),holder,Task,reach,task,total,tasks,windows>:
  Windowed{+model,+packed,+holds} = windowed
  Windowed{model,packed,windows_projected(~Element,~Source,~value,~Holder,~project,~Task,~reach,holder,task,total,tasks,model,holds)}

def windows_hold(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+limits: Machine.Limits,+grid: Grid.Grid,+layout: Held.Spans,+total: Nat,+laid: Laid<limits,grid,layout,total>,+task: Task,+source: Source,+fill: Element,+limit: U32,+limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},+pieces_ok: {U32.is_le(Grid.pieces_of(grid),2147483648) == True{} : Bool},+tasks: Pieces.Ranges,+reaches: Reaches(~Task,~reach,task,tasks,total),pieces: Pieces.Pieces(~Element),+holding: Held.Holding<Element>,
  +packed: {Move.held(~Element,pieces) == Held.pack(~Element,holding) : Move.Holding<Element>},+facts: HeldFacts<Element,Source,value,source,holding,total,layout>,
  +nonempty: {Held.spanned(Held.spans(~Element,Held.order(~Element,holding))) == True{} : Bool}) ->
  Windowed<Element,Source,value,source,Task,reach,task,total,tasks,Halo.windows(~Element,~Task,~reach,task,grid,fill,tasks,pieces)>:
  +walked = walk_model(~Element,~Task,~reach,task,grid,fill,tasks,holding,0)
  Windowed{windows_model(~Element,~Task,~reach,task,grid,fill,tasks,holding),
    windows_packed(~Element,~Task,~reach,task,grid,fill,tasks,pieces,holding,packed,nonempty),
    handling_windows(~Element,~Source,~value,~Task,~reach,task,grid,total,source,tasks,0,Held.order(~Element,holding),
      handled_model(~Element,walked_halos(~Element,walked),Held.order(~Element,walked_holding(~Element,walked)),0),
      tasks_step(~Element,~Source,~value,~Task,~reach,limits,grid,layout,total,laid,task,source,fill,limit,limit_ok,small,pieces_ok,tasks,reaches,holding,0,Held.order(~Element,walked_holding(~Element,walked)),0,facts,Order.zero_le(U32.to_nat(Halo.piece_from(grid,0))),Order.zero_le(total),
        walk_order(~Element,~Task,~reach,task,grid,fill,tasks,holding,0)))}

# Pieces whose holding covers the cells of a layout in order give their windows.
def covered_windows(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Task: Data,~reach: Halo.Reach(~Task),+limits: Machine.Limits,+grid: Grid.Grid,
  +ranges: Pieces.Ranges,+total: Nat,+laid: Laid<limits,grid,Held.layout_spans(ranges,Held.Ended{}),total>,+task: Task,+source: Source,+fill: Element,+limit: U32,
  +limit_ok: {U32.is_le(limit,1073741824) == True{} : Bool},+small: {Nat.is_le(total,U32.to_nat(limit)) == True{} : Bool},
  +pieces_ok: {U32.is_le(Grid.pieces_of(grid),2147483648) == True{} : Bool},+tasks: Pieces.Ranges,+reaches: Reaches(~Task,~reach,task,tasks,total),
  pieces: Pieces.Pieces(~Element),+covered: Held.Covered<Element,Source,value,source,Move.held(~Element,pieces),0n,total,Held.layout_spans(ranges,Held.Ended{})>) ->
  Windowed<Element,Source,value,source,Task,reach,task,total,tasks,Halo.windows(~Element,~Task,~reach,task,grid,fill,tasks,pieces)>:
  Held.Covered{+model,+packed,+tiled,+values,+ordered} = covered
  windows_hold(~Element,~Source,~value,~Task,~reach,limits,grid,Held.layout_spans(ranges,Held.Ended{}),total,laid,task,source,fill,limit,limit_ok,small,pieces_ok,
    tasks,reaches,pieces,model,packed,HeldFacts{tiled,values,ordered},
    {%Equal.sym(Held.Spans,Held.spans(~Element,Held.order(~Element,model)),Held.layout_spans(ranges,Held.Ended{}),ordered) : {Held.spanned(_) == True{} : Bool}
      Held.layout_spanned(ranges,Held.Ended{}) : {Held.spanned(Held.spans(~Element,Held.order(~Element,model))) == True{} : Bool}})

def pieces_of_tensor(~Element: Data,tensor: Tensors.Tensor<Element>) -> Pieces.Pieces(~Element):
  Tensors.Tensor{pieces,shape,fill,status,workers} = tensor
  pieces

def reduced_covered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+model: PiecesModel.Model<Element>,+contract: KernelGemm.Both<KernelGemm.Both<Content.WellFormed<Element,ShapeLaws.elements(shape),PiecesModel.contents(~Element,PiecesModel.pack(~Element,model))>,KernelGemm.Both<Global.Counted<shape>,Global.Canonical(~Element,shape,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)))>>,Content.Values(~Element,~Logical,~original,logical,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)))>) -> Held.Covered<Element,Logical,original,logical,Move.held(~Element,PiecesModel.pack(~Element,model)),0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>:
  KernelGemm.Both{+normal,+values} = contract
  KernelGemm.Both{+formed,+rest} = normal
  KernelGemm.Both{+counted,+canonical} = rest
  +kept = PiecesModel.packed_contents(~Element,model)
  +formed_model = {%kept : Content.WellFormed<Element,ShapeLaws.elements(shape),_>
    formed : Content.WellFormed<Element,ShapeLaws.elements(shape),PiecesModel.content(~Element,model)>}
  +values_model = {%kept : Content.Values(~Element,~Logical,~original,logical,_)
    values : Content.Values(~Element,~Logical,~original,logical,PiecesModel.content(~Element,model))}
  +laid = Equal.trans(Pieces.Ranges,PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),PiecesModel.content_layout(~Element,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model))),Pieces.ranges_of_grid(Tensors.grid_of(shape)),
    Equal.cong(PiecesModel.Content<Element>,Pieces.Ranges,content => PiecesModel.content_layout(~Element,content),PiecesModel.content(~Element,model),PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),Equal.sym(PiecesModel.Content<Element>,PiecesModel.contents(~Element,PiecesModel.pack(~Element,model)),PiecesModel.content(~Element,model),kept)),
    canonical)
  %laid :
    Held.Covered<Element,Logical,original,logical,Move.held(~Element,PiecesModel.pack(~Element,model)),0n,ShapeLaws.elements(shape),Held.layout_spans(_,Held.Ended{})>
  Held.held_covered(~Element,~Logical,~original,logical,model,ShapeLaws.elements(shape),formed_model,values_model)

def model_covered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+model: PiecesModel.Model<Element>,+model_fill: Element,+model_workers: U32,-pieces: Pieces.Pieces(~Element),
  +fill: Element,+status: Tensors.Status,+workers: U32,+equation: {Tensors.Tensor{PiecesModel.pack(~Element,model),shape,model_fill,Tensors.Usable{},model_workers} == Tensors.Tensor{pieces,shape,fill,status,workers} : Tensors.Tensor<Element>},
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,status,workers})) -> Held.Covered<Element,Logical,original,logical,Move.held(~Element,pieces),0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>:
  +packed = Equal.cong(Tensors.Tensor<Element>,Pieces.Pieces(~Element),tensor => pieces_of_tensor(~Element,tensor),Tensors.Tensor{PiecesModel.pack(~Element,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,status,workers},equation)
  +reduced = {%Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,status,workers},equation) : Global.Contract(~Element,~Logical,~original,logical,_)
    contract : Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,model_fill,Tensors.Usable{},model_workers})}
  %packed : Held.Covered<Element,Logical,original,logical,Move.held(~Element,_),0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>
  reduced_covered(~Element,~Logical,~original,logical,shape,model,reduced)

# The pieces of a Shaped tensor cover its cells in the order of its grid.
def shaped_covered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,-pieces: Pieces.Pieces(~Element),+fill: Element,+status: Tensors.Status,+workers: U32,
  +shaped: Global.Shaped(~Element,~Logical,~original,logical,shape,Tensors.Tensor{pieces,shape,fill,status,workers})) -> Held.Covered<Element,Logical,original,logical,Move.held(~Element,pieces),0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>:
  KernelGemm.Both{+usable,+contract} = shaped
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  model_covered(~Element,~Logical,~original,logical,shape,model,model_fill,model_workers,pieces,fill,status,workers,equation,contract)
