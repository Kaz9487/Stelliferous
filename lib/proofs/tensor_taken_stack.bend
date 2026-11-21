# The stack of the pieces a component takes of a source. Its pieces rise
# through the grid and each ends before the next starts, so the stack lies in
# order within the cells.
import Base
import ../tensor_grid.bend as Grid
import ../kernel_gemm.bend as KernelGemm
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_task_runs.bend as Runs
import ./tensor_moved.bend as Moved
import ./tensor_components.bend as Components
import ./tensor_grid_pieces.bend as Laws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_grid_box.bend as Box
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ../tensor_tasks.bend as Tasks
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Logic

# ---- Where pieces end ----

# A piece ends where the next piece starts.
def piece_end(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+U32.to_nat(piece)))) : Nat}:
  %Arithmetic.native_roundtrip(piece) :
    {Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},_))) == U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(1n+U32.to_nat(piece)))) : Nat}
  GridRanges.piece_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.to_nat(piece),inside)

# A piece ends at or before every later piece starts.
def end_before(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,+later: U32,
  +later_in: {Nat.is_le(U32.to_nat(later),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool},
  +before: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(later)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},later))) == True{} : Bool}:
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +next = {1n+U32.to_nat(piece) : Nat}
  +next_le = Order.strict_follow(U32.to_nat(piece),U32.to_nat(later),before)
  +room = Order.transitive(next,total,U32.to_nat(Machine.capacity(limits)),Order.transitive(next,U32.to_nat(later),total,next_le,later_in),
    Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +back = Bounds.roundtrip(next,Machine.capacity(limits),LimitBounds.capacity_fits(limits),room)
  %Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(next))),
      piece_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,Box.lt_le(U32.to_nat(piece),U32.to_nat(later),total,before,later_in))) :
    {Nat.is_le(_,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},later))) == True{} : Bool}
  Laws.start_monotone(limits,outer,extent,unit_cells,block,per,count,prefix,shape,U32.from_nat(next),later,later_in,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(next)),next,back) : {Nat.is_le(_,U32.to_nat(later)) == True{} : Bool} next_le :
      {Nat.is_le(U32.to_nat(U32.from_nat(next)),U32.to_nat(later)) == True{} : Bool}})

# A piece ends within the cells.
def end_within(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+piece: U32,
  +inside: {Nat.is_lt(U32.to_nat(piece),Nat.mul(U32.to_nat(outer),U32.to_nat(per))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))),U32.to_nat(count)) == True{} : Bool}:
  +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
  +next = {1n+U32.to_nat(piece) : Nat}
  +next_le = Order.strict_follow(U32.to_nat(piece),total,inside)
  +room = Order.transitive(next,total,U32.to_nat(Machine.capacity(limits)),next_le,Laws.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  +back = Bounds.roundtrip(next,Machine.capacity(limits),LimitBounds.capacity_fits(limits),room)
  %Equal.sym(Nat,Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece))),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},U32.from_nat(next))),
      piece_end(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside)) :
    {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
  GridRanges.start_cells(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.from_nat(next),
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(next)),next,back) : {Nat.is_le(_,total) == True{} : Bool} next_le :
      {Nat.is_le(U32.to_nat(U32.from_nat(next)),total) == True{} : Bool}})

# ---- The stack in order ----

def spans_first(spans: Held.Spans) -> U32:
  match spans:
    case Held.Ended{}: 0
    case Held.Spanned{first,count,rest}: first

def spans_count(spans: Held.Spans) -> U32:
  match spans:
    case Held.Ended{}: 0
    case Held.Spanned{first,count,rest}: count

def spans_rest(spans: Held.Spans) -> Held.Spans:
  match spans:
    case Held.Ended{}: Held.Ended{}
    case Held.Spanned{first,count,rest}: rest

def spanned(spans: Held.Spans) -> Bool:
  match spans:
    case Held.Ended{}: False{}
    case Held.Spanned{first,count,rest}: True{}

def ended_spanned(+first: U32,+count: U32,+rest: Held.Spans,+same: {Held.Ended{} == Held.Spanned{first,count,rest} : Held.Spans}) -> Empty:
  WordEncoding.false_true(Equal.cong(Held.Spans,Bool,spanned,Held.Ended{},Held.Spanned{first,count,rest},same))

def spanned_ended(+first: U32,+count: U32,+rest: Held.Spans,+same: {Held.Spanned{first,count,rest} == Held.Ended{} : Held.Spans}) -> Empty:
  ended_spanned(first,count,rest,Equal.sym(Held.Spans,Held.Spanned{first,count,rest},Held.Ended{},same))

# Every held piece holds as many cells as it counts.
def StackSized(~Element: Data,stack: Held.Stack<Element>) -> Data:
  match stack:
    case Held.Bottom{}: Unit
    case Held.Held{+buffer,+first,+count,below}: Held.Both<{Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},StackSized(~Element,below)>

def stacked_sized(~Element: Data,+model: PiecesModel.Model<Element>,+rest: Held.Stack<Element>,
  +sized: PiecesModel.ContentAll(~Element,~Runs.piece_sized(~Element),PiecesModel.content(~Element,model)),+after: StackSized(~Element,rest)) ->
  StackSized(~Element,Held.stacked_model(~Element,model,rest)):
  match model:
    case PiecesModel.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+count}: Held.Both{sized,after}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_sized,+right_sized} = sized
      stacked_sized(~Element,left,Held.stacked_model(~Element,right,rest),left_sized,stacked_sized(~Element,right,rest,right_sized,after))

# The first cell the next taken piece starts at or after.
def StartsFrom(pieces: List<&2,U32>,+grid: Grid.Grid,+low: Nat) -> Data:
  match pieces:
    case Nil{}: {Nat.is_le(low,0n) == True{} : Bool}
    case Con{+piece,rest}: {Nat.is_le(low,U32.to_nat(Grid.start(grid,piece))) == True{} : Bool}

# The stack of the pieces a component takes, rising through the grid, lies
# in order within the cells.
def taken_sorted(~Element: Data,+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+count: U32,+prefix: U32,
  +shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  stack: Held.Stack<Element>,pieces: List<&2,U32>,+low: Nat,+from: Nat,+laid: {Held.spans(~Element,stack) == Moved.taken_spans(pieces,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}) : Held.Spans},
  +sized: StackSized(~Element,stack),+rising: Components.Rising(pieces,from,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),+starts: StartsFrom(pieces,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},low)) ->
  Held.Sorted(~Element,stack,low,U32.to_nat(count)):
  match stack pieces:
    case Held.Held{+buffer,+first,+size,Held.Bottom{}} Nil{}:
      Held.Both{+size_ok,+later_sized} = sized
      +whole = Held.spans(~Element,Held.Held{buffer,first,size,Held.Bottom{}})
      +taken = Moved.taken_spans(Nil{},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})
      +first_zero = Equal.cong(Held.Spans,U32,spans_first,whole,taken,laid)
      +size_zero = Equal.cong(Held.Spans,U32,spans_count,whole,taken,laid)
      Held.Both{Held.From{{%Equal.sym(U32,first,0,first_zero) : {Nat.is_le(low,U32.to_nat(_)) == True{} : Bool} starts : {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool}},
          size_ok},
        {%Equal.sym(U32,first,0,first_zero) : {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(size)),U32.to_nat(count)) == True{} : Bool}
          %Equal.sym(U32,size,0,size_zero) : {Nat.is_le(Nat.add(0n,U32.to_nat(_)),U32.to_nat(count)) == True{} : Bool}
          Order.zero_le(U32.to_nat(count)) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(size)),U32.to_nat(count)) == True{} : Bool}}}
    case Held.Held{+buffer,+first,+size,Held.Held{+next_buffer,+next_first,+next_size,+rest}} Nil{}:
      +whole = Held.spans(~Element,Held.Held{buffer,first,size,Held.Held{next_buffer,next_first,next_size,rest}})
      Empty.absurd(Held.Sorted(~Element,Held.Held{buffer,first,size,Held.Held{next_buffer,next_first,next_size,rest}},low,U32.to_nat(count)),
        spanned_ended(next_first,next_size,Held.spans(~Element,rest),Equal.cong(Held.Spans,Held.Spans,spans_rest,whole,Moved.taken_spans(Nil{},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),laid)))
    case Held.Held{+buffer,+first,+size,Held.Bottom{}} Con{+piece,Nil{}}:
      Held.Both{+size_ok,+later_sized} = sized
      Held.Both{+at,+more} = rising
      Held.Both{+inside,+later} = more
      +whole = Held.spans(~Element,Held.Held{buffer,first,size,Held.Bottom{}})
      +taken = Moved.taken_spans(Con{piece,Nil{}},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})
      +first_eq = Equal.cong(Held.Spans,U32,spans_first,whole,taken,laid)
      +size_eq = Equal.cong(Held.Spans,U32,spans_count,whole,taken,laid)
      Held.Both{Held.From{{%Equal.sym(U32,first,Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),first_eq) : {Nat.is_le(low,U32.to_nat(_)) == True{} : Bool} starts :
            {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool}},size_ok},
        {%Equal.sym(U32,first,Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),first_eq) : {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(size)),U32.to_nat(count)) == True{} : Bool}
          %Equal.sym(U32,size,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),size_eq) : {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(_)),U32.to_nat(count)) == True{} : Bool}
          end_within(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,inside) :
          {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(size)),U32.to_nat(count)) == True{} : Bool}}}
    case Held.Held{+buffer,+first,+size,Held.Held{+next_buffer,+next_first,+next_size,+rest}} Con{+piece,Nil{}}:
      +whole = Held.spans(~Element,Held.Held{buffer,first,size,Held.Held{next_buffer,next_first,next_size,rest}})
      Empty.absurd(Held.Sorted(~Element,Held.Held{buffer,first,size,Held.Held{next_buffer,next_first,next_size,rest}},low,U32.to_nat(count)),
        spanned_ended(next_first,next_size,Held.spans(~Element,rest),Equal.cong(Held.Spans,Held.Spans,spans_rest,whole,Moved.taken_spans(Con{piece,Nil{}},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),laid)))
    case Held.Held{+buffer,+first,+size,+below} Con{+piece,Con{+next,+more}}:
      Held.Both{+size_ok,+later_sized} = sized
      Held.Both{+at,+rest_rising} = rising
      Held.Both{+inside,+later} = rest_rising
      Held.Both{+next_at,+next_more} = later
      Held.Both{+next_inside,+next_later} = next_more
      +total = Nat.mul(U32.to_nat(outer),U32.to_nat(per))
      +whole = Held.spans(~Element,Held.Held{buffer,first,size,below})
      +taken = Moved.taken_spans(Con{piece,Con{next,more}},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)})
      +first_eq = Equal.cong(Held.Spans,U32,spans_first,whole,taken,laid)
      +size_eq = Equal.cong(Held.Spans,U32,spans_count,whole,taken,laid)
      +rest_eq = Equal.cong(Held.Spans,Held.Spans,spans_rest,whole,taken,laid)
      +end = Nat.add(U32.to_nat(first),U32.to_nat(size))
      +next_starts = {%Equal.sym(U32,first,Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),first_eq) :
          {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(size)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},next))) == True{} : Bool}
        %Equal.sym(U32,size,Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),size_eq) :
          {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),U32.to_nat(_)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},next))) == True{} : Bool}
        end_before(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,piece,next,Bounds.strict_le(U32.to_nat(next),total,next_inside),
          Order.succ_le_lt(U32.to_nat(piece),U32.to_nat(next),next_at)) :
        {Nat.is_le(end,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},next))) == True{} : Bool}}
      Held.Both{Held.From{{%Equal.sym(U32,first,Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),first_eq) : {Nat.is_le(low,U32.to_nat(_)) == True{} : Bool} starts :
            {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool}},size_ok},
        taken_sorted(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,below,Con{next,more},end,1n+U32.to_nat(piece),rest_eq,later_sized,
          Held.Both{next_at,Held.Both{next_inside,next_later}},next_starts)}
    case Held.Bottom{} Nil{}:
      Empty.absurd(Held.Sorted(~Element,Held.Bottom{},low,U32.to_nat(count)),ended_spanned(0,0,Held.Ended{},laid))
    case Held.Bottom{} Con{+piece,Nil{}}:
      Empty.absurd(Held.Sorted(~Element,Held.Bottom{},low,U32.to_nat(count)),ended_spanned(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),Held.Ended{},laid))
    case Held.Bottom{} Con{+piece,Con{+next,+more}}:
      Empty.absurd(Held.Sorted(~Element,Held.Bottom{},low,U32.to_nat(count)),
        ended_spanned(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),Grid.span(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece),Moved.taken_spans(Con{next,more},Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),laid))

# ---- The pieces a component of one source row takes ----

def starts_zero(pieces: List<&2,U32>,+grid: Grid.Grid) -> StartsFrom(pieces,grid,0n):
  match pieces:
    case Nil{}: {==}
    case Con{+piece,+rest}: Order.zero_le(U32.to_nat(Grid.start(grid,piece)))

def row_rising_separate(~Extra: Data,+limits: Machine.Limits,+source: List<&2,U32>,+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(source,Grid.piece_target()) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
  +room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +separate: Bool,+yes: {separate == True{} : Bool},
  +facts: Held.Both<Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(row,classes),Extra>>) ->
  Components.Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.box_when(separate,row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,offset)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  Held.Both{Held.Both{+rooms,Held.Both{+axes_ok,+axes_below}},Held.Both{+ends,+extra}} = facts
  %Equal.sym(Bool,separate,True{},yes) :
    Components.Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.box_when(_,row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,offset)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  %Equal.sym(List<&2,U32>,Grid.coordinates(source,offset),Laws.coordinates_model(source,offset),Laws.coordinates_exact(source,offset)) :
    Components.Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.boxed(row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,_,0)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  Components.source_rising(limits,source,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,row,classes,offset,rooms,ends,
    {%exact : Components.AxesOk(row,_,source,0,Machine.half_word(limits)) axes_ok : Components.AxesOk(row,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,0,Machine.half_word(limits))})

def row_rising_whole(+limits: Machine.Limits,+source: List<&2,U32>,+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: Laws.Shape<limits,outer,extent,unit_cells,block,per>,
  +exact: {Grid.grid(source,Grid.piece_target()) == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},+shape_ok: Components.ShapeOk(limits,source),
  +separate: Bool,+whole: {separate == False{} : Bool}) ->
  Components.Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.box_when(separate,row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,offset)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))):
  %Equal.sym(Bool,separate,False{},whole) :
    Components.Rising(Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.box_when(_,row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,offset)),0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per)))
  Components.whole_rising(limits,source,Grid.piece_target(),count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok)

def row_sorted_facts(~Element: Data,~Extra: Data,+limits: Machine.Limits,+source: List<&2,U32>,+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +separate: Bool,+model: PiecesModel.Model<Element>,+facts: Laws.GridFacts<limits,Grid.grid(source,Grid.piece_target())>,+shape_ok: Components.ShapeOk(limits,source),
  +separate_facts: Logic.When(separate,Held.Both<Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(row,classes),Extra>>),
  +sized: PiecesModel.ContentAll(~Element,~Runs.piece_sized(~Element),PiecesModel.content(~Element,model)),
  +laid: {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) ==
    Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.box_when(separate,row,classes,Grid.grid(source,Grid.piece_target()),source,offset)),
      Grid.grid(source,Grid.piece_target())) : Held.Spans}) ->
  Held.Sorted(~Element,Held.stacked_model(~Element,model,Held.Bottom{}),0n,ShapeLaws.elements(source)):
  Laws.GridFacts{+count,+prefix,+outer,+extent,+unit_cells,+block,+per,+shape,+exact,+cells} = facts
  Held.Both{+room,Held.Both{+positive,+rank_room}} = shape_ok
  +stack = Held.stacked_model(~Element,model,Held.Bottom{})
  +pieces = Tasks.box_pieces(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,Tasks.box_when(separate,row,classes,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},source,offset))
  +flat_laid = {%exact : {Held.spans(~Element,stack) == Moved.taken_spans(Tasks.box_pieces(_,source,Tasks.box_when(separate,row,classes,_,source,offset)),_) : Held.Spans}
    laid : {Held.spans(~Element,stack) == Moved.taken_spans(pieces,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}) : Held.Spans}}
  +count_exact = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(Grid.count_of(Grid.grid(source,Grid.piece_target()))),ShapeLaws.elements(source),
    Equal.cong(Grid.Grid,Nat,found => U32.to_nat(Grid.count_of(found)),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Grid.grid(source,Grid.piece_target()),Equal.sym(Grid.Grid,Grid.grid(source,Grid.piece_target()),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},exact)),
    GridRanges.grid_count(limits,source,Grid.piece_target(),room,positive))
  +rising = Held.by_cases(Components.Rising(pieces,0n,Nat.mul(U32.to_nat(outer),U32.to_nat(per))),separate,
    +yes => row_rising_separate(~Extra,limits,source,row,classes,offset,count,prefix,outer,extent,unit_cells,block,per,shape,exact,room,positive,rank_room,separate,yes,
      {%yes : Logic.When(_,Held.Both<Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(row,classes),Extra>>) separate_facts :
        Held.Both<Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(row,classes),Extra>>}),
    +no => row_rising_whole(limits,source,row,classes,offset,count,prefix,outer,extent,unit_cells,block,per,shape,exact,shape_ok,separate,no))
  %count_exact : Held.Sorted(~Element,stack,0n,_)
  taken_sorted(~Element,limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,stack,pieces,0n,0n,flat_laid,stacked_sized(~Element,model,Held.Bottom{},sized,Unit{}),rising,
    starts_zero(pieces,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}))

# The pieces a component of one source row took, stacked as the component
# took them, lie in order within the cells of the source.
def row_sorted(~Element: Data,~Extra: Data,+limits: Machine.Limits,+source: List<&2,U32>,+row: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+offset: U32,
  +separate: Bool,+model: PiecesModel.Model<Element>,+shape_ok: Components.ShapeOk(limits,source),
  +separate_facts: Logic.When(separate,Held.Both<Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source),Held.Both<Components.RowEnds(row,classes),Extra>>),
  +sized: PiecesModel.ContentAll(~Element,~Runs.piece_sized(~Element),PiecesModel.content(~Element,model)),
  +laid: {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) ==
    Moved.taken_spans(Tasks.box_pieces(Grid.grid(source,Grid.piece_target()),source,Tasks.box_when(separate,row,classes,Grid.grid(source,Grid.piece_target()),source,offset)),
      Grid.grid(source,Grid.piece_target())) : Held.Spans}) ->
  Held.Sorted(~Element,Held.stacked_model(~Element,model,Held.Bottom{}),0n,ShapeLaws.elements(source)):
  Held.Both{+room,Held.Both{+positive,+rank_room}} = shape_ok
  row_sorted_facts(~Element,~Extra,limits,source,row,classes,offset,separate,model,GridRanges.grid_facts(limits,source,Grid.piece_target(),room,positive),shape_ok,separate_facts,sized,laid)
