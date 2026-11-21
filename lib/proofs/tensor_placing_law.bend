# The law of a placing: every target piece of a component starts as fill and
# takes from every source the cells the segments of the source place, each
# source holding the values of its own logical tensor.

import Base
import ./storage_buffer_model.bend as Buffer
import ./tensor_placed.bend as Placed
import ./word_encoding.bend as WordEncoding
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../tensor.bend as Tensors
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_move_readback.bend as Held
import ./tensor_moved.bend as Moved
import ./tensor_move_parts.bend as MoveParts
import ./tensor_placements.bend as Placements
import ./tensor_placing_parts.bend as Parts
import ./tensor_placing_reads.bend as Reads
import ./tensor_placing_gather.bend as Gathered
import ./tensor_taken_stack.bend as Taken
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_task_runs.bend as Runs
import ./tensor_components.bend as Components
import ./tensor_target_cells.bend as Targets
import ./tensor_view_reads.bend as ViewReads
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Logic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_view_gather.bend as ViewGather
import ./storage_tree_model.bend as Tree
import ./natural_addition.bend as Addition
import ./word_subtraction.bend as Subtraction
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_box_digits.bend as Digits

# ---- What every source holds ----

# The logicals after the first count of them.
def dropped(~Logical: Data,count: Nat,list: List<&2,Logical>) -> List<&2,Logical>:
  match count:
    case 0n: list
    case 1n+ +more:
      match list:
        case Nil{}: Nil{}
        case Con{+logical,+rest}: dropped(~Logical,more,rest)

# A piece holds the values of the logical at position at.
def HeldFrom(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,at: Nat,logicals: List<&2,Logical>,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  match at:
    case 0n:
      match logicals:
        case Nil{}: Unit
        case Con{+logical,+rest}: Placed.PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)
    case 1n+ +more:
      match logicals:
        case Nil{}: Unit
        case Con{+logical,+rest}: HeldFrom(~Element,~Logical,~original,more,rest,first,rows,buffer)

# A piece of source source holds the values of the logical of the source.
def HeldAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+source: U32,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  HeldFrom(~Element,~Logical,~original,U32.to_nat(source),logicals,first,rows,buffer)

# A piece that holds the logical at position count holds the first logical
# after count of them.
def held_dropped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,count: Nat,+logicals: List<&2,Logical>,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  held: HeldFrom(~Element,~Logical,~original,count,logicals,first,rows,buffer)) ->
  HeldFrom(~Element,~Logical,~original,0n,dropped(~Logical,count,logicals),first,rows,buffer):
  match count:
    case 0n: held
    case 1n+ +more:
      match logicals:
        case Nil{}: Unit{}
        case Con{+logical,+rest}: held_dropped(~Element,~Logical,~original,more,rest,first,rows,buffer,held)

# The list after its first entry.
def after_first(~Logical: Data,list: List<&2,Logical>) -> List<&2,Logical>:
  match list:
    case Nil{}: Nil{}
    case Con{+logical,+rest}: rest

def is_cons(~Logical: Data,list: List<&2,Logical>) -> Bool:
  match list:
    case Nil{}: False{}
    case Con{+logical,+rest}: True{}

# An empty list is no list with an entry.
def list_apart(~Logical: Data,+logical: Logical,+rest: List<&2,Logical>,+same: {Nil{} == Con{logical,rest} : List<&2,Logical>}) -> Empty:
  WordEncoding.false_true(Equal.cong(List<&2,Logical>,Bool,values => is_cons(~Logical,values),Nil{},Con{logical,rest},same))

# The logicals after one more are the rest of the logicals after count.
def dropped_next(~Logical: Data,count: Nat,list: List<&2,Logical>,+logical: Logical,+rest: List<&2,Logical>,
  +at: {dropped(~Logical,count,list) == Con{logical,rest} : List<&2,Logical>}) -> {dropped(~Logical,1n+count,list) == rest : List<&2,Logical>}:
  match count:
    case 0n:
      match list:
        case Nil{}: Empty.absurd({dropped(~Logical,1n,Nil{}) == rest : List<&2,Logical>},list_apart(~Logical,logical,rest,at))
        case Con{+head,+tail}: Equal.cong(List<&2,Logical>,List<&2,Logical>,values => after_first(~Logical,values),Con{head,tail},Con{logical,rest},at)
    case 1n+ +more:
      match list:
        case Nil{}: Empty.absurd({dropped(~Logical,1n+(1n+more),Nil{}) == rest : List<&2,Logical>},list_apart(~Logical,logical,rest,at))
        case Con{+head,+tail}: dropped_next(~Logical,more,tail,logical,rest,at)

# ---- What every source needs ----

# A source of a placing is ready to place its parts in a target piece: it fits
# half a word, its holding is in order and holds its values, its parts are
# ready, and its axes fit the piece.
def source_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+own: List<&2,List<&2,Tasks.Segment>>,+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},+word: U32,+bounded: Digits.StridesIn(Tasks.strides_of(source),word),+capacity: U32,+holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},
  +placed: Held.Both<Placements.AxesIn(own,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Tasks.strides_of(source),Machine.half_word(limits)),Parts.AxesReading(own,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),clamped,Machine.half_word(limits))>,
  +holding: Held.Holding<Element>,+model: PiecesModel.Model<Element>,+ordered: {Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},
  +held: PiecesModel.ContentAll(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),PiecesModel.content(~Element,model)),
  +laid: {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans}) ->
  Gathered.SourceReady(~Element,~Logical,~original,logical,own,source,holding,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),word,capacity):
  Held.Both{+axes_in,+axes} = placed
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +cap_half = Bounds.as_nat_le(Machine.capacity(limits),cap,Limits.capacity_within_half(limits))
  +rank_cap = Order.transitive(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),rank_room,cap_half)
  +row_separate = Reads.placed_separable(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,cap,cap_ok,rank_cap)
  +laid_true = {%separate : {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(_,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans} laid : {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(True{},Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans}}
  +laid_row = {%Equal.sym(Bool,Tasks.separable([Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)]),True{},row_separate) : {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(_,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans} laid_true :
    {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable([Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)]),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans}}
  +separate_when = {%Equal.sym(Bool,Tasks.separable([Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)]),True{},row_separate) :
      Logic.When(_,Held.Both<Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),Held.Both<Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),Unit>>)
    Held.Both{row_ok,Held.Both{ends,Unit{}}} :
    Logic.When(Tasks.separable([Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)]),Held.Both<Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),Held.Both<Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),Unit>>)}
  +sized = PiecesModel.content_map(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),~Runs.piece_sized(~Element),~Placed.holds_sized(~Element,~Logical,~original,logical),
    PiecesModel.content(~Element,model),held)
  +sorted = Taken.row_sorted(~Element,~Unit,limits,source,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),0,
    Tasks.separable([Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)]),model,Held.Both{source_room,Held.Both{source_nonempty,source_rank}},separate_when,sized,laid_row)
  +spans_same = Equal.trans(Held.Spans,Held.spans(~Element,Held.order(~Element,holding)),Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})),Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),
    Equal.cong(Held.Stack<Element>,Held.Spans,stack => Held.spans(~Element,stack),Held.order(~Element,holding),Held.stacked_model(~Element,model,Held.Bottom{}),ordered),laid)
  Held.Both{Order.transitive(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),source_room,cap_half),
    Held.Both{{%Equal.sym(Held.Stack<Element>,Held.order(~Element,holding),Held.stacked_model(~Element,model,Held.Bottom{}),ordered) : Held.Sorted(~Element,_,0n,ShapeLaws.elements(source)) sorted :
        Held.Sorted(~Element,Held.order(~Element,holding),0n,ShapeLaws.elements(source))},
      Held.Both{Placed.ordered_values(~Element,~Logical,~original,logical,model,holding,ordered,held),
        Held.Both{{%Equal.sym(Held.Spans,Held.spans(~Element,Held.order(~Element,holding)),Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),spans_same) : MoveParts.PartsReady(Tasks.placements(own,source,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece)),cap,word,_,capacity)
            Parts.placements_ready(own,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,word,bounded,capacity,holds,cells_exact,axes_in,axes) : MoveParts.PartsReady(Tasks.placements(own,source,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece)),cap,word,Held.spans(~Element,Held.order(~Element,holding)),capacity)},axes_in}}}}

# A source of a placing is ready to place its parts in a target piece whose
# axes fit half a word from its low corner: its shape fits the machine, its
# readings fit and read its axes, its blocks fit and its axes are placed.
def source_placed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+own: List<&2,List<&2,Tasks.Segment>>,+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},+classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),+capacity: U32,+holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},+rooms: Digits.Rooms(Digits.words_nat(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece))),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Machine.half_word(limits)),+source_ok: Components.ShapeOk(limits,source),+row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),+blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+source_axes: Parts.SourceAxes(own,target,source,shifts,clamped,Machine.half_word(limits)),+holding: Held.Holding<Element>,+model: PiecesModel.Model<Element>,+ordered: {Held.order(~Element,holding) == Held.stacked_model(~Element,model,Held.Bottom{}) : Held.Stack<Element>},+held: PiecesModel.ContentAll(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),PiecesModel.content(~Element,model)),+laid: {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans}) ->
  Gathered.SourceReady(~Element,~Logical,~original,logical,own,source,holding,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity):
  Held.Both{+source_room,Held.Both{+source_nonempty,+source_rank}} = source_ok
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank_cap = Order.transitive(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),rank_room,
    Bounds.as_nat_le(Machine.capacity(limits),cap,Limits.capacity_within_half(limits)))
  source_ready(~Element,~Logical,~original,logical,own,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,Reads.placed_distinct(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0,cap,cap_ok,rank_cap),cap,Digits.strides_in(source,limits,source_room,source_nonempty),capacity,holds,cells_exact,Parts.axes_placed(own,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),clamped,cap,rooms,source_axes,Digits.coordinates_count(target,Grid.start(Tensors.grid_of(target),piece)),Digits.box_of_length(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Parts.strides_length(source),Parts.strides_length(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)))),holding,model,ordered,held,laid)

# ---- Source after source ----

# The word before a positive word.
def before_word(+word: U32,+count: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(word) == 1n+count : Nat},
  +room: {Nat.is_le(1n+count,U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((word - 1 : U32)) == count : Nat}:
  +count_room = Order.transitive(count,1n+count,U32.to_nat(cap),Order.successor(count,count,Order.reflexive(count)),room)
  +back = Bounds.roundtrip(count,cap,cap_ok,count_room)
  +step = U32.from_nat(count)
  +sum_room = {%Equal.sym(Nat,Nat.add(count,1n),1n+count,Addition.successor_width(count)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(Nat.add(count,1n),U32.to_nat(cap)) == True{} : Bool}}
  +sum = Equal.trans(Nat,U32.to_nat((step + 1 : U32)),Nat.add(count,1n),1n+count,Arithmetic.sum_exact(cap,cap_ok,step,1,count,1n,back,{==},sum_room),Addition.successor_width(count))
  +same = Arithmetic.to_nat_injective(word,(step + 1 : U32),
    Equal.trans(Nat,U32.to_nat(word),1n+count,U32.to_nat((step + 1 : U32)),exact,Equal.sym(Nat,U32.to_nat((step + 1 : U32)),1n+count,sum)))
  %Equal.sym(U32,word,(step + 1 : U32),same) : {U32.to_nat((_ - 1 : U32)) == count : Nat}
  %Equal.sym(U32,((step + 1 : U32) - 1 : U32),step,Subtraction.subtract_add(step,1)) : {U32.to_nat(_) == count : Nat}
  back

# The readings at a source are the first readings after as many as it
# counts.
def readings_dropped(count: Nat,rows: List<&2,List<&2,Tasks.Reading>>,+source: U32,+row: List<&2,Tasks.Reading>,+rest: List<&2,List<&2,Tasks.Reading>>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+exact: {U32.to_nat(source) == count : Nat},+room: {Nat.is_le(count,U32.to_nat(cap)) == True{} : Bool},
  +at: {dropped(~List<&2,Tasks.Reading>,count,rows) == Con{row,rest} : List<&2,List<&2,Tasks.Reading>>}) -> {Components.readings_word(rows,source) == row : List<&2,Tasks.Reading>}:
  match count rows:
    case 0n Nil{}:
      Empty.absurd({Components.readings_word(Nil{},source) == row : List<&2,Tasks.Reading>},list_apart(~List<&2,Tasks.Reading>,row,rest,at))
    case 0n Con{+own,+more}:
      %Equal.sym(U32,source,0,Arithmetic.to_nat_injective(source,0,exact)) : {Components.readings_word(Con{own,more},_) == row : List<&2,Tasks.Reading>}
      Equal.cong(List<&2,List<&2,Tasks.Reading>>,List<&2,Tasks.Reading>,values => Components.readings_word(values,0),Con{own,more},Con{row,rest},at)
    case 1n+ +m Nil{}:
      Empty.absurd({Components.readings_word(Nil{},source) == row : List<&2,Tasks.Reading>},list_apart(~List<&2,Tasks.Reading>,row,rest,at))
    case 1n+ +m Con{+own,+more}:
      +nonzero = Arithmetic.word_positive_nonzero(source,{%Equal.sym(Nat,U32.to_nat(source),1n+m,exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,U32.to_nat(source)) == True{} : Bool}})
      %Equal.sym(Bool,U32.is_zero(source),False{},nonzero) : {Bool.pick(List<&2,Tasks.Reading>,_,own,Components.readings_word(more,(source - 1 : U32))) == row : List<&2,Tasks.Reading>}
      readings_dropped(m,more,(source - 1 : U32),row,rest,cap,cap_ok,before_word(source,m,cap,cap_ok,exact,room),
        Order.transitive(m,1n+m,U32.to_nat(cap),Order.successor(m,m,Order.reflexive(m)),room),at)

# A piece that holds the first logical holds the values of its source.
def held_first(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+rest: List<&2,Logical>,+first: U32,+rows: U32,
  +buffer: Buffer.Content<Element>,held: HeldFrom(~Element,~Logical,~original,0n,Con{logical,rest},first,rows,buffer)) ->
  Placed.SourceHolds(~Element,~Logical,~original,logical,0,first,rows,buffer):
  held

# The readings of every source placed in a target.
def placing_rows(+target: List<&2,U32>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>) -> List<&2,List<&2,Tasks.Reading>>:
  match sources shifts clamps:
    case Con{+source,sources} Con{+shift,shifts} Con{+clamped,clamps}:
      Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0) <> placing_rows(target,sources,shifts,clamps)
    case sources shifts clamps: []

# The origins of the sources placed in a target.
def placing_origins(sources: List<&2,List<&2,U32>>) -> List<&2,Tensors.Origin>:
  match sources:
    case Nil{}: []
    case Con{+source,sources}: Tensors.origin(source,0) <> placing_origins(sources)

# What every source placed in a target needs: its shape fits the machine,
# its readings fit half a word and read its axes, its blocks fit, and its
# axes are placed; the lists end together.
def SourcesFacts(+limits: Machine.Limits,+target: List<&2,U32>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,
  shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>) -> Data:
  match segments sources shifts clamps:
    case Con{own,segments} Con{+source,sources} Con{+shift,shifts} Con{+clamped,clamps}:
      Held.Both<Components.ShapeOk(limits,source),
        Held.Both<Components.RowOk(limits,Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),
          Held.Both<ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),
            Held.Both<Parts.SourceAxes(own,target,source,shift,clamped,Machine.half_word(limits)),SourcesFacts(limits,target,segments,sources,shifts,clamps)>>>>
    case Nil{} Nil{} Nil{} Nil{}:
      Unit
    case segments sources shifts clamps:
      Empty

# Source after source from source_index on, every source of a placing is
# ready to place its parts in a target piece: the logicals, the readings and
# the holdings of the sources follow the sources.
def sources_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,
  clamps: List<&2,Bool>,logicals: List<&2,Logical>,models: Moved.Models<Element>,holdings: Moved.Holdings<Element>,+all: List<&2,Logical>,+readings: List<&2,List<&2,Tasks.Reading>>,
  +source_index: U32,+limits: Machine.Limits,+target: List<&2,U32>,+component: U32,+piece: U32,+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},+classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),+capacity: U32,+holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},+rooms: Digits.Rooms(Digits.words_nat(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece))),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Machine.half_word(limits)),
  +facts: SourcesFacts(limits,target,segments,sources,shifts,clamps),+ends: Components.AllEnds(placing_rows(target,sources,shifts,clamps),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +held: Moved.SourcesHeld(~Element,~HeldAt(~Element,~Logical,~original,all),placing_origins(sources),source_index,models),
  +laid: Moved.SourcesLaid(~Element,placing_origins(sources),Tasks.components_of(readings),source_index,component,models),
  +order: Moved.HeldOrder(~Element,holdings,models),+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +at_logicals: {dropped(~Logical,U32.to_nat(source_index),all) == logicals : List<&2,Logical>},
  +at_rows: {dropped(~List<&2,Tasks.Reading>,U32.to_nat(source_index),readings) == placing_rows(target,sources,shifts,clamps) : List<&2,List<&2,Tasks.Reading>>},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,placing_rows(target,sources,shifts,clamps))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Gathered.PlacedReady(~Element,~Logical,~original,logicals,segments,sources,holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity):
  match segments sources shifts clamps:
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      match logicals:
        case Con{+logical,+logicals}:
          match models holdings:
            case Moved.Models{+model,+more} Moved.Holds{+holding,+rest}:
              Held.Both{+source_ok,Held.Both{+row_ok,Held.Both{+blocks,Held.Both{+source_axes,+facts_rest}}}} = facts
              Held.Both{+row_ends,+ends_rest} = ends
              Held.Both{+held_here,+held_rest} = held
              Held.Both{+laid_here,+laid_rest} = laid
              Held.Both{+ordered,+order_rest} = order
              +cap = Machine.half_word(limits)
              +cap_ok = Limits.half_word_fits(limits)
              +at = U32.to_nat(source_index)
              +row = Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)
              +later = placing_rows(target,sources,shifts,clamps)
              +more_count = List.length(&2,List<&2,Tasks.Reading>,later)
              +next = Reads.next_axis(source_index,more_count,cap,cap_ok,count_room)
              +at_room = Order.transitive(at,Nat.add(at,1n+more_count),U32.to_nat(cap),Order.le_add(at,1n+more_count),count_room)
              +row_word = readings_dropped(at,readings,source_index,row,later,cap,cap_ok,{==},at_room,at_rows)
              +row_at = Equal.trans(List<&2,Tasks.Reading>,Tasks.readings_at(readings,at),Components.readings_word(readings,source_index),row,
                Components.readings_word_exact(readings,source_index,cap,cap_ok,at_room),row_word)
              +laid_row = {%row_at : {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),_,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans} laid_here : {Held.spans(~Element,Held.stacked_model(~Element,model,Held.Bottom{})) == Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),row,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)) : Held.Spans}}
              +dropped_held = PiecesModel.content_map(~Element,~HeldAt(~Element,~Logical,~original,all,source_index),~HeldFrom(~Element,~Logical,~original,0n,dropped(~Logical,at,all)),
                ~held_dropped(~Element,~Logical,~original,at,all),PiecesModel.content(~Element,model),held_here)
              +first_held = {%at_logicals : PiecesModel.ContentAll(~Element,~HeldFrom(~Element,~Logical,~original,0n,_),PiecesModel.content(~Element,model)) dropped_held :
                PiecesModel.ContentAll(~Element,~HeldFrom(~Element,~Logical,~original,0n,Con{logical,logicals}),PiecesModel.content(~Element,model))}
              +source_held = PiecesModel.content_map(~Element,~HeldFrom(~Element,~Logical,~original,0n,Con{logical,logicals}),~Placed.SourceHolds(~Element,~Logical,~original,logical,0),
                ~held_first(~Element,~Logical,~original,logical,logicals),PiecesModel.content(~Element,model),first_held)
              +after_logicals = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {dropped(~Logical,_,all) == logicals : List<&2,Logical>}
                dropped_next(~Logical,at,all,logical,logicals,at_logicals) : {dropped(~Logical,U32.to_nat((source_index + 1 : U32)),all) == logicals : List<&2,Logical>}}
              +after_rows = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {dropped(~List<&2,Tasks.Reading>,_,readings) == later : List<&2,List<&2,Tasks.Reading>>}
                dropped_next(~List<&2,Tasks.Reading>,at,readings,row,later,at_rows) : {dropped(~List<&2,Tasks.Reading>,U32.to_nat((source_index + 1 : U32)),readings) == later : List<&2,List<&2,Tasks.Reading>>}}
              +after_room = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {Nat.is_le(Nat.add(_,more_count),U32.to_nat(cap)) == True{} : Bool}
                Order.advance_room(at,more_count,U32.to_nat(cap),count_room) : {Nat.is_le(Nat.add(U32.to_nat((source_index + 1 : U32)),more_count),U32.to_nat(cap)) == True{} : Bool}}
              +named_rest = Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Logical,logicals),1n+List.length(&2,List<&2,U32>,sources),named)
              Held.Both{source_placed(~Element,~Logical,~original,logical,own,limits,target,source,shift,clamped,readings,component,piece,room,nonempty,rank_room,inside,
                  separate,classes_ok,member,capacity,holds,cells_exact,rooms,source_ok,row_ok,row_ends,blocks,source_axes,holding,model,ordered,source_held,laid_row),
                sources_ready(~Element,~Logical,~original,segments,sources,shifts,clamps,logicals,more,rest,all,readings,(source_index + 1 : U32),limits,target,component,piece,room,nonempty,rank_room,inside,separate,classes_ok,member,capacity,holds,cells_exact,rooms,
                  facts_rest,ends_rest,held_rest,laid_rest,order_rest,named_rest,after_logicals,after_rows,after_room)}
            case Moved.NoModels{} +holdings:
              Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,Con{logical,logicals},Con{own,segments},Con{source,sources},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),held)
            case Moved.Models{+model,+more} Moved.Unheld{}:
              Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,Con{logical,logicals},Con{own,segments},Con{source,sources},Moved.Unheld{},Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),order)
        case Nil{}:
          Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,Nil{},Con{own,segments},Con{source,sources},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),Digits.zero_successor(List.length(&2,List<&2,U32>,sources),named))
    case Nil{} Nil{} Nil{} Nil{}:
      match logicals:
        case Nil{}: Unit{}
        case Con{+logical,+logicals}:
          Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,Con{logical,logicals},Nil{},Nil{},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),
            Digits.zero_successor(List.length(&2,Logical,logicals),Equal.sym(Nat,1n+List.length(&2,Logical,logicals),0n,named)))
    case Con{+own,+segments} Nil{} +shifts +clamps:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Con{own,segments},Nil{},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)
    case Con{+own,+segments} Con{+source,+sources} Nil{} +clamps:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Con{own,segments},Con{source,sources},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Nil{}:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Con{own,segments},Con{source,sources},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)
    case Nil{} Con{+source,+sources} +shifts +clamps:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Nil{},Con{source,sources},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)
    case Nil{} Nil{} Con{+shift,+shifts} +clamps:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Nil{},Nil{},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)
    case Nil{} Nil{} Nil{} Con{+clamped,+clamps}:
      Empty.absurd(Gathered.PlacedReady(~Element,~Logical,~original,logicals,Nil{},Nil{},holdings,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),Machine.half_word(limits),Machine.half_word(limits),capacity),facts)

# ---- The classes of a placing ----

# Limits for every target axis of rank axes are as many as the cells.
def limits_rank(rank: Nat,cells: List<&2,U32>,+readings: List<&2,List<&2,Tasks.Reading>>,+grid: Grid.Grid,+at: U32,
  +limits_at: Components.LimitsAt(readings,rank,cells,grid,at)) -> {rank == List.length(&2,U32,cells) : Nat}:
  match rank cells:
    case 1n+ +rest Con{+extent,+more}:
      Held.Both{+here,+later} = limits_at
      Equal.cong(Nat,Nat,count => 1n+count,rest,List.length(&2,U32,more),limits_rank(rest,more,Tasks.tails(readings),grid,(at + 1 : U32),later))
    case 0n Nil{}:
      {==}
    case 0n Con{+extent,+more}:
      Empty.absurd({0n == List.length(&2,U32,Con{extent,more}) : Nat},limits_at)
    case 1n+ +rest Nil{}:
      Empty.absurd({1n+rest == List.length(&2,U32,Nil{}) : Nat},limits_at)

# The classes a placed source reads every target axis with end inside its
# blocks, and the axis with a block fits cap.
def placed_class_ends(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,classes: List<&2,Tasks.Class>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,
  +clamped: Bool,+axis: U32,+cap: U32,+same: {List.length(&2,Tasks.Class,classes) == List.length(&2,U32,target) : Nat},
  +full: {List.length(&2,Tasks.Reading,Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis)) == List.length(&2,U32,target) : Nat},
  +rooms: Components.ReadingsRoom(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis),cap),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis),classes)) ->
  Targets.ClassEnds(classes,target_grid,target,axis,cap):
  match target source shifts classes:
    case Con{+extent,+target} Con{+bound,+source} Con{+shift,+shifts} Con{+class,+classes}:
      Held.Both{Held.Both{Held.Both{+blocks_exact,Held.Both{+ceiling_room,Held.Both{+block_positive,+sum_room}}},+read_room},+rooms_rest} = rooms
      Held.Both{+end_in,+ends_rest} = ends
      Held.Both{end_in,Held.Both{sum_room,placed_class_ends(target,source,shifts,classes,target_grid,source_grid,clamped,(axis + 1 : U32),cap,
        Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Tasks.Class,classes),1n+List.length(&2,U32,target),same),
        Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Tasks.Reading,Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32))),1n+List.length(&2,U32,target),full),rooms_rest,ends_rest)}}
    case Con{+extent,+target} Nil{} +shifts +classes:
      Empty.absurd(Targets.ClassEnds(classes,target_grid,Con{extent,target},axis,cap),Digits.zero_successor(List.length(&2,U32,target),full))
    case Con{+extent,+target} Con{+bound,+source} Nil{} +classes:
      Empty.absurd(Targets.ClassEnds(classes,target_grid,Con{extent,target},axis,cap),Digits.zero_successor(List.length(&2,U32,target),full))
    case Con{+extent,+target} Con{+bound,+source} Con{+shift,+shifts} Nil{}:
      Empty.absurd(Targets.ClassEnds(Nil{},target_grid,Con{extent,target},axis,cap),Digits.zero_successor(List.length(&2,U32,target),same))
    case Nil{} +source +shifts Nil{}:
      Unit{}
    case Nil{} +source +shifts Con{+class,+classes}:
      Empty.absurd(Targets.ClassEnds(Con{class,classes},target_grid,Nil{},axis,cap),
        Digits.zero_successor(List.length(&2,Tasks.Class,classes),Equal.sym(Nat,1n+List.length(&2,Tasks.Class,classes),0n,same)))

# No class of no reading ends anywhere, and a target of no axes needs none.
def no_classes(target: List<&2,U32>,+grid: Grid.Grid,+cap: U32,+component: U32,+empty: {0n == List.length(&2,U32,target) : Nat}) ->
  Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(0n,[]),component),grid,target,0,cap):
  match target:
    case Nil{}: Unit{}
    case Con{+extent,+more}: Empty.absurd(Targets.ClassEnds([],grid,Con{extent,more},0,cap),Digits.zero_successor(List.length(&2,U32,more),empty))

# The classes of a component of a placing end inside the blocks of every
# target axis.
def classes_ends(segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>,+limits: Machine.Limits,+target: List<&2,U32>,+component: U32,
  +facts: SourcesFacts(limits,target,segments,sources,shifts,clamps),+ends: Components.AllEnds(placing_rows(target,sources,shifts,clamps),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),placing_rows(target,sources,shifts,clamps)),component)),
  +limits_at: Components.LimitsAt(placing_rows(target,sources,shifts,clamps),Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),target,Tensors.grid_of(target),0)) -> Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),placing_rows(target,sources,shifts,clamps)),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)):
  match segments sources shifts clamps:
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      Held.Both{+source_ok,Held.Both{Held.Both{+rooms,+row_more},+facts_rest}} = facts
      Held.Both{+row_ends,+ends_rest} = ends
      +rows = {Con{Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),placing_rows(target,sources,shifts,clamps)} : List<&2,List<&2,Tasks.Reading>>}
      +rank_exact = limits_rank(Tasks.target_rank(rows),target,rows,Tensors.grid_of(target),0,limits_at)
      placed_class_ends(target,source,shift,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(rows),rows),component),Tensors.grid_of(target),Tensors.grid_of(source),clamped,0,Machine.half_word(limits),
        Equal.trans(Nat,List.length(&2,Tasks.Class,Tasks.chosen(Tasks.class_lists(Tasks.target_rank(rows),rows),component)),Tasks.target_rank(rows),List.length(&2,U32,target),
          Targets.chosen_rank(Tasks.target_rank(rows),rows,component),rank_exact),rank_exact,rooms,row_ends)
    case Nil{} Nil{} Nil{} Nil{}:
      no_classes(target,Tensors.grid_of(target),Machine.half_word(limits),component,limits_rank(0n,target,[],Tensors.grid_of(target),0,limits_at))
    case Con{+own,+segments} Nil{} +shifts +clamps:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Nil{},shifts,clamps)),placing_rows(target,Nil{},shifts,clamps)),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)
    case Con{+own,+segments} Con{+source,+sources} Nil{} +clamps:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Con{source,sources},Nil{},clamps)),placing_rows(target,Con{source,sources},Nil{},clamps)),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Nil{}:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Con{source,sources},Con{shift,shifts},Nil{})),placing_rows(target,Con{source,sources},Con{shift,shifts},Nil{})),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)
    case Nil{} Con{+source,+sources} +shifts +clamps:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Con{source,sources},shifts,clamps)),placing_rows(target,Con{source,sources},shifts,clamps)),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)
    case Nil{} Nil{} Con{+shift,+shifts} +clamps:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Nil{},Con{shift,shifts},clamps)),placing_rows(target,Nil{},Con{shift,shifts},clamps)),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)
    case Nil{} Nil{} Nil{} Con{+clamped,+clamps}:
      Empty.absurd(Targets.ClassEnds(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(placing_rows(target,Nil{},Nil{},Con{clamped,clamps})),placing_rows(target,Nil{},Nil{},Con{clamped,clamps})),component),Tensors.grid_of(target),target,0,Machine.half_word(limits)),facts)

# Every source placed in a target reads every axis once and none across.
def rows_separable(+target: List<&2,U32>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(cap)) == True{} : Bool}) -> {Tasks.separable(placing_rows(target,sources,shifts,clamps)) == True{} : Bool}:
  match sources shifts clamps:
    case Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      %Equal.sym(Bool,Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))),True{},Reads.placed_distinct(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0,cap,cap_ok,room)) :
        {((_ && Bool.not(Tasks.crossing(Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)))) && Tasks.separable(placing_rows(target,sources,shifts,clamps))) == True{} : Bool}
      %Equal.sym(Bool,Tasks.crossing(Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)),False{},Reads.placed_uncrossed(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)) :
        {((True{} && Bool.not(_)) && Tasks.separable(placing_rows(target,sources,shifts,clamps))) == True{} : Bool}
      rows_separable(target,sources,shifts,clamps,cap,cap_ok,room)
    case Nil{} +shifts +clamps: {==}
    case Con{+source,+sources} Nil{} +clamps: {==}
    case Con{+source,+sources} Con{+shift,+shifts} Nil{}: {==}

# Components that fit cap read no more rows than cap.
def rows_room_of(+readings: List<&2,List<&2,Tasks.Reading>>,+rank: Nat,+cap: U32,+ok: Components.ComponentsOk(readings,rank,cap)) ->
  {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,readings),U32.to_nat(cap)) == True{} : Bool}:
  Held.Both{+rows,Held.Both{+columns,Held.Both{+lists_room,Held.Both{+room,+rows_room}}}} = ok
  rows_room

# ---- The law of a placing ----

# What a placing needs: its target fits the machine, the components of its
# readings fit half a word with limits for every target axis, and every
# source has what it needs.
def PlacingFacts(+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>) -> Data:
  Held.Both<Components.ShapeOk(limits,target),
    Held.Both<Components.ComponentsOk(placing_rows(target,sources,shifts,clamps),Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),Machine.half_word(limits)),
      Held.Both<{Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),placing_rows(target,sources,shifts,clamps)))) == True{} : Bool},
        Held.Both<Components.LimitsAt(placing_rows(target,sources,shifts,clamps),Tasks.target_rank(placing_rows(target,sources,shifts,clamps)),target,Tensors.grid_of(target),0),SourcesFacts(limits,target,segments,sources,shifts,clamps)>>>>

# A gather context agrees with a placing: it places the sources in the target
# through the components of their readings, from their origins, with the
# fill of the placing, and the placing has what it needs.
def PlacingAgrees(~Element: Data,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+filled: Element,+context: Tensors.Placing,
  +components: Tasks.Components,+origins: List<&2,Tensors.Origin>,+cells: List<&2,U32>,+cells_grid: Grid.Grid,+fill: Element) -> Data:
  Held.Both<{context == Tensors.Placing{Tensors.grid_of(target),target,sources,segments} : Tensors.Placing},
  Held.Both<{components == Tasks.components_of(placing_rows(target,sources,shifts,clamps)) : Tasks.Components},
  Held.Both<{origins == placing_origins(sources) : List<&2,Tensors.Origin>},
  Held.Both<{cells == target : List<&2,U32>},
  Held.Both<{cells_grid == Tensors.grid_of(target) : Grid.Grid},
  Held.Both<{fill == filled : Element},PlacingFacts(limits,target,segments,sources,shifts,clamps)>>>>>>

def placing_law_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+filled: Element,
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: PlacingFacts(limits,target,segments,sources,shifts,clamps),
  +index: U32,+piece: U32,+models: Moved.Models<Element>,+holdings: Moved.Holdings<Element>,
  +ready: Moved.TakenReady(~Element,~HeldAt(~Element,~Logical,~original,logicals),Tasks.components_of(placing_rows(target,sources,shifts,clamps)),placing_origins(sources),models,index),
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components_of(placing_rows(target,sources,shifts,clamps))))) == True{} : Bool},+order: Moved.HeldOrder(~Element,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Tensors.grid_of(target),target,index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),filled,holdings,models,
    Gathered.placing_model(~Element,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),filled,holdings)):
  Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+sources_facts}}}} = facts
  Held.Both{+held,+laid} = ready
  +readings = placing_rows(target,sources,shifts,clamps)
  +rank = Tasks.target_rank(readings)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank_cap = Order.transitive(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),rank_room,
    Bounds.as_nat_le(Machine.capacity(limits),cap,Limits.capacity_within_half(limits)))
  +component_in = Targets.component_lists(limits,readings,components_ok,lists_positive,index,inside)
  +separate = rows_separable(target,sources,shifts,clamps,cap,cap_ok,rank_cap)
  +ends = Components.runtime_ends(readings,rank,cap,cap_ok,components_ok,index,component_in)
  +classes_ok = Targets.classes_ok(limits,target,Grid.piece_target(),room,nonempty,rank_room,readings,index,components_ok,limits_at,component_in,
    classes_ends(segments,sources,shifts,clamps,limits,target,index,sources_facts,ends,limits_at))
  +piece_in = Targets.target_piece_below(limits,target,Grid.piece_target(),room,nonempty,rank_room,readings,index,
    {%Equal.sym(Bool,Tasks.separable(readings),True{},separate) : Logic.When(_,Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),index),Tensors.grid_of(target),target,cap)) classes_ok :
      Logic.When(Tasks.separable(readings),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),index),Tensors.grid_of(target),target,cap))},piece,member)
  +cells_exact = Targets.grid_box_cells(limits,target,Grid.piece_target(),room,nonempty,rank_room,piece,piece_in)
  +fresh = Tree.fresh(Element,Tensors.depth_for(Grid.span(Tensors.grid_of(target),piece)),filled)
  +holds = {%Equal.sym(Nat,ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(Grid.span(Tensors.grid_of(target),piece)),cells_exact) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}
    Gathered.fresh_holds(~Element,limits,target,piece,filled,room,nonempty,piece_in) : {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  +all_ready = sources_ready(~Element,~Logical,~original,segments,sources,shifts,clamps,logicals,models,holdings,logicals,readings,0,limits,target,index,piece,room,nonempty,rank_room,
    piece_in,separate,classes_ok,member,Tree.capacity(Element,fresh),holds,cells_exact,Digits.grid_box_rooms(limits,target,Grid.piece_target(),room,nonempty,rank_room,piece,piece_in),
    sources_facts,ends,held,laid,order,named,{==},{==},rows_room_of(readings,rank,cap,components_ok))
  Held.Both{Gathered.placing_written(~Element,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),filled,holdings),
    Held.Both{Gathered.placed_order(~Element,segments,sources,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),holdings,models,ViewGather.fresh_output(~Element,filled,Grid.span(Tensors.grid_of(target),piece)),order),
      Gathered.placing_target(~Element,~Logical,~original,logicals,segments,sources,holdings,limits,target,piece,filled,cap,cap_ok,room,nonempty,rank_room,piece_in,all_ready)}}

# A placing gathers every target piece of its components with the values the
# sources give its cells from the fill, as its model says, keeping the
# holdings in order.
def placing_law(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+filled: Element,
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+components: Tasks.Components,+origins: List<&2,Tensors.Origin>,
  +cells: List<&2,U32>,+cells_grid: Grid.Grid,+fill: Element,+context: Tensors.Placing,+index: U32,+piece: U32,+models: Moved.Models<Element>,+holdings: Moved.Holdings<Element>,
  +agrees: PlacingAgrees(~Element,limits,target,segments,sources,shifts,clamps,filled,context,components,origins,cells,cells_grid,fill),
  +ready: Moved.TakenReady(~Element,~HeldAt(~Element,~Logical,~original,logicals),components,origins,models,index),
  +inside: {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(components))) == True{} : Bool},+order: Moved.HeldOrder(~Element,holdings,models),
  +member: Moved.Member(Tasks.target_pieces(components,cells_grid,cells,index),piece)) ->
  Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,Gathered.placing_model(~Element,context,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings)):
  Held.Both{+same_context,Held.Both{+same_components,Held.Both{+same_origins,Held.Both{+same_cells,Held.Both{+same_grid,Held.Both{+same_fill,+facts}}}}}} = agrees
  %Equal.sym(Tensors.Placing,context,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},same_context) : Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),_,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,Gathered.placing_model(~Element,_,components,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Tasks.Components,components,Tasks.components_of(placing_rows(target,sources,shifts,clamps)),same_components) : Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),Tensors.Placing{Tensors.grid_of(target),target,sources,segments},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings,models,Gathered.placing_model(~Element,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},_,Grid.start(cells_grid,piece),Grid.span(cells_grid,piece),fill,holdings))
  %Equal.sym(Grid.Grid,cells_grid,Tensors.grid_of(target),same_grid) : Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(_,piece),Grid.span(_,piece),fill,holdings,models,Gathered.placing_model(~Element,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(_,piece),Grid.span(_,piece),fill,holdings))
  %Equal.sym(Element,fill,filled,same_fill) : Moved.GatherFacts(~Element,~Tensors.Placing,~Tensors.placing_gather(~Element),~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,filled}),Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),_,holdings,models,Gathered.placing_model(~Element,Tensors.Placing{Tensors.grid_of(target),target,sources,segments},Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece),_,holdings))
  placing_law_at(~Element,~Logical,~original,logicals,limits,target,segments,sources,shifts,clamps,filled,named,facts,index,piece,models,holdings,
    {%same_origins : Moved.TakenReady(~Element,~HeldAt(~Element,~Logical,~original,logicals),Tasks.components_of(placing_rows(target,sources,shifts,clamps)),_,models,index)
      %same_components : Moved.TakenReady(~Element,~HeldAt(~Element,~Logical,~original,logicals),_,origins,models,index)
      ready : Moved.TakenReady(~Element,~HeldAt(~Element,~Logical,~original,logicals),Tasks.components_of(placing_rows(target,sources,shifts,clamps)),placing_origins(sources),models,index)},
    {%same_components : {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(_))) == True{} : Bool} inside : {Nat.is_lt(U32.to_nat(index),U32.to_nat(Tasks.components_count(Tasks.components_of(placing_rows(target,sources,shifts,clamps))))) == True{} : Bool}},order,
    {%same_cells : Moved.Member(Tasks.target_pieces(Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Tensors.grid_of(target),_,index),piece)
      %same_grid : Moved.Member(Tasks.target_pieces(Tasks.components_of(placing_rows(target,sources,shifts,clamps)),_,cells,index),piece)
      %same_components : Moved.Member(Tasks.target_pieces(_,cells_grid,cells,index),piece)
      member : Moved.Member(Tasks.target_pieces(Tasks.components_of(placing_rows(target,sources,shifts,clamps)),Tensors.grid_of(target),target,index),piece)})
