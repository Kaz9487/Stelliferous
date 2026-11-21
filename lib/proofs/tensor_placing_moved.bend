# The tensor a placing writes: sources of the contract placed in a target
# give a tensor of the contract that holds, in every target cell, the value
# the sources give the cell from the fill.

import Base
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_layout.bend as Layout
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./machine_limit_bounds.bend as Limits
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_moved_sources.bend as Sources
import ./tensor_then_gather.bend as Then
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_global_shape_count.bend as Global
import ./tensor_placing_reads.bend as Reads
import ./tensor_placing_gather.bend as Gathered
import ./tensor_placing_law.bend as Law
import ./tensor_box_digits.bend as Digits
import ./boolean_logic.bend as Logic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds

# ---- The sources ----

# Every source of a placing is laid as the canonical pieces of its shape and
# every piece holds the values of its logical; the lists end together.
def SourcesPlaced(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,logicals: List<&2,Logical>,sources: List<&2,List<&2,U32>>,models: Moved.Models<Element>) -> Data:
  match logicals sources models:
    case Con{+logical,logicals} Con{+source,sources} Moved.Models{+model,models}:
      Held.Both<{PiecesModel.layout(~Element,model) == Pieces.ranges_of_grid(Tensors.grid_of(source)) : Pieces.Ranges},
        Held.Both<PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model)),SourcesPlaced(~Element,~Logical,~original,logicals,sources,models)>>
    case Nil{} Nil{} Moved.NoModels{}:
      Unit
    case logicals sources models:
      Empty

# A piece that holds the values of the first logical after count holds the
# values of the logical at count.
def held_undropped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,count: Nat,logicals: List<&2,Logical>,+logical: Logical,+rest: List<&2,Logical>,
  +at: {Law.dropped(~Logical,count,logicals) == Con{logical,rest} : List<&2,Logical>},+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  held: Placed.PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)) -> Law.HeldFrom(~Element,~Logical,~original,count,logicals,first,rows,buffer):
  match count logicals:
    case 0n Nil{}:
      Empty.absurd(Law.HeldFrom(~Element,~Logical,~original,0n,Nil{},first,rows,buffer),Law.list_apart(~Logical,logical,rest,at))
    case 0n Con{+head,+tail}:
      %Equal.sym(List<&2,Logical>,Con{head,tail},Con{logical,rest},at) : Law.HeldFrom(~Element,~Logical,~original,0n,_,first,rows,buffer)
      held
    case 1n+ +more Nil{}:
      Empty.absurd(Law.HeldFrom(~Element,~Logical,~original,1n+more,Nil{},first,rows,buffer),Law.list_apart(~Logical,logical,rest,at))
    case 1n+ +more Con{+head,+tail}:
      held_undropped(~Element,~Logical,~original,more,tail,logical,rest,at,first,rows,buffer,held)

# A list that has an entry after count entries is longer than count.
def dropped_below(~Logical: Data,count: Nat,list: List<&2,Logical>,+logical: Logical,+rest: List<&2,Logical>,
  +at: {Law.dropped(~Logical,count,list) == Con{logical,rest} : List<&2,Logical>}) -> {Nat.is_lt(count,List.length(&2,Logical,list)) == True{} : Bool}:
  match count list:
    case 0n Nil{}:
      Empty.absurd({Nat.is_lt(0n,List.length(&2,Logical,Nil{})) == True{} : Bool},Law.list_apart(~Logical,logical,rest,at))
    case 0n Con{+head,+tail}:
      {==}
    case 1n+ +more Nil{}:
      Empty.absurd({Nat.is_lt(1n+more,List.length(&2,Logical,Nil{})) == True{} : Bool},Law.list_apart(~Logical,logical,rest,at))
    case 1n+ +more Con{+head,+tail}:
      dropped_below(~Logical,more,tail,logical,rest,at)

# The shapes of the origins of the sources are the sources.
def shapes_of(sources: List<&2,List<&2,U32>>) -> {Tensors.origin_shapes(Law.placing_origins(sources)) == sources : List<&2,List<&2,U32>>}:
  match sources:
    case Nil{}: {==}
    case Con{+source,+rest}:
      Equal.cong(List<&2,List<&2,U32>>,List<&2,List<&2,U32>>,shapes => source <> shapes,Tensors.origin_shapes(Law.placing_origins(rest)),rest,shapes_of(rest))

def walk_joined(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+all: List<&2,Logical>,+readings: List<&2,List<&2,Tasks.Reading>>,+source_index: U32,+limits: Machine.Limits,+fill: Element,+components: Tasks.Components,
  +source: List<&2,U32>,+sources: List<&2,List<&2,U32>>,+model: PiecesModel.Model<Element>,+models: Moved.Models<Element>,
  +row_when: Logic.When(Tasks.separable(readings),Held.Both<{Nat.is_lt(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},Components.RowOk(limits,Components.readings_word(readings,source_index),Moved.origin_grid(Tensors.origin(source,0)),Components.origin_axes(Tensors.origin(source,0)))>),+source_ok: Components.ShapeOk(limits,source),
  +laid: {PiecesModel.layout(~Element,model) == Pieces.ranges_of_grid(Tensors.grid_of(source)) : Pieces.Ranges},
  +held_empty: Law.HeldAt(~Element,~Logical,~original,all,source_index,0,0,Buffer.content(~Element,Moved.empty(~Element,fill))),
  +good: Placed.Good(~Element,~Law.HeldAt(~Element,~Logical,~original,all,source_index),model,Components.gone_upto(components,source_index,Tensors.origin(source,0),0n)),
  +later: Held.Both<Components.OriginsOk(limits,Law.placing_origins(sources),readings,(source_index + 1 : U32),Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(sources),(source_index + 1 : U32),fill,models,Components.gones_upto(Law.placing_origins(sources),components,(source_index + 1 : U32),0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(sources),(source_index + 1 : U32),models,Components.gones_upto(Law.placing_origins(sources),components,(source_index + 1 : U32),0n))>>) ->
  Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,Moved.Models{model,models},Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,Moved.Models{model,models},Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>:
  Held.Both{+ok_rest,Held.Both{+static_rest,+good_rest}} = later
  Held.Both{Held.Both{Held.Both{{==},Held.Both{{==},source_ok}},Held.Both{row_when,ok_rest}},
    Held.Both{Held.Both{laid,Held.Both{Components.shape_ordered(limits,source,source_ok),Held.Both{held_empty,static_rest}}},Held.Both{good,good_rest}}}

# Source after source from source_index on, every source of a placing is laid
# as its origin says and reads its row, and is static and good before
# anything is taken.
def sources_moved(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>,logicals: List<&2,Logical>,models: Moved.Models<Element>,+all: List<&2,Logical>,+readings: List<&2,List<&2,Tasks.Reading>>,+source_index: U32,
  +limits: Machine.Limits,+target: List<&2,U32>,+fill: Element,+components: Tasks.Components,+separate: {Tasks.separable(readings) == True{} : Bool},
  +facts: Law.SourcesFacts(limits,target,segments,sources,shifts,clamps),+placed: SourcesPlaced(~Element,~Logical,~original,logicals,sources,models),
  +named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},
  +at_logicals: {Law.dropped(~Logical,U32.to_nat(source_index),all) == logicals : List<&2,Logical>},
  +at_rows: {Law.dropped(~List<&2,Tasks.Reading>,U32.to_nat(source_index),readings) == Law.placing_rows(target,sources,shifts,clamps) : List<&2,List<&2,Tasks.Reading>>},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,sources,shifts,clamps))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Held.Both<Components.OriginsOk(limits,Law.placing_origins(sources),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(sources),source_index,fill,models,Components.gones_upto(Law.placing_origins(sources),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(sources),source_index,models,Components.gones_upto(Law.placing_origins(sources),components,source_index,0n))>>:
  match segments sources shifts clamps:
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      match logicals:
        case Con{+logical,+logicals}:
          match models:
            case Moved.Models{+model,+models}:
              Held.Both{+source_ok,Held.Both{+row_ok,Held.Both{+blocks,Held.Both{+source_axes,+facts_rest}}}} = facts
              Held.Both{+laid,Held.Both{+holds,+placed_rest}} = placed
              +cap = Machine.half_word(limits)
              +cap_ok = Limits.half_word_fits(limits)
              +at = U32.to_nat(source_index)
              +row = Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0)
              +later = Law.placing_rows(target,sources,shifts,clamps)
              +more_count = List.length(&2,List<&2,Tasks.Reading>,later)
              +next = Reads.next_axis(source_index,more_count,cap,cap_ok,count_room)
              +row_at = Law.readings_dropped(at,readings,source_index,row,later,cap,cap_ok,{==},
                Order.transitive(at,Nat.add(at,1n+more_count),U32.to_nat(cap),Order.le_add(at,1n+more_count),count_room),at_rows)
              +row_facts = {Held.Both{dropped_below(~List<&2,Tasks.Reading>,at,readings,row,later,at_rows),
                  {%Equal.sym(List<&2,Tasks.Reading>,Components.readings_word(readings,source_index),row,row_at) : Components.RowOk(limits,_,Tensors.grid_of(source),source)
                    row_ok : Components.RowOk(limits,Components.readings_word(readings,source_index),Tensors.grid_of(source),source)}} : Held.Both<{Nat.is_lt(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},Components.RowOk(limits,Components.readings_word(readings,source_index),Moved.origin_grid(Tensors.origin(source,0)),Components.origin_axes(Tensors.origin(source,0)))>}
              +row_when = {%Equal.sym(Bool,Tasks.separable(readings),True{},separate) : Logic.When(_,Held.Both<{Nat.is_lt(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},Components.RowOk(limits,Components.readings_word(readings,source_index),Moved.origin_grid(Tensors.origin(source,0)),Components.origin_axes(Tensors.origin(source,0)))>) row_facts : Logic.When(Tasks.separable(readings),Held.Both<{Nat.is_lt(U32.to_nat(source_index),List.length(&2,List<&2,Tasks.Reading>,readings)) == True{} : Bool},Components.RowOk(limits,Components.readings_word(readings,source_index),Moved.origin_grid(Tensors.origin(source,0)),Components.origin_axes(Tensors.origin(source,0)))>)}
              +held_empty = held_undropped(~Element,~Logical,~original,at,all,logical,logicals,at_logicals,0,0,Buffer.content(~Element,Moved.empty(~Element,fill)),
                Sources.empty_holds(~Element,~Logical,~original,logical,fill))
              +holds_at = PiecesModel.content_map(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),~Law.HeldAt(~Element,~Logical,~original,all,source_index),
                ~held_undropped(~Element,~Logical,~original,at,all,logical,logicals,at_logicals),PiecesModel.content(~Element,model),holds)
              +good = Sources.good_all(~Element,~Law.HeldAt(~Element,~Logical,~original,all,source_index),model,Components.gone_upto(components,source_index,Tensors.origin(source,0),0n),holds_at)
              +after_logicals = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {Law.dropped(~Logical,_,all) == logicals : List<&2,Logical>}
                Law.dropped_next(~Logical,at,all,logical,logicals,at_logicals) : {Law.dropped(~Logical,U32.to_nat((source_index + 1 : U32)),all) == logicals : List<&2,Logical>}}
              +after_rows = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {Law.dropped(~List<&2,Tasks.Reading>,_,readings) == later : List<&2,List<&2,Tasks.Reading>>}
                Law.dropped_next(~List<&2,Tasks.Reading>,at,readings,row,later,at_rows) : {Law.dropped(~List<&2,Tasks.Reading>,U32.to_nat((source_index + 1 : U32)),readings) == later : List<&2,List<&2,Tasks.Reading>>}}
              +after_room = {%Equal.sym(Nat,U32.to_nat((source_index + 1 : U32)),1n+at,next) : {Nat.is_le(Nat.add(_,more_count),U32.to_nat(cap)) == True{} : Bool}
                Order.advance_room(at,more_count,U32.to_nat(cap),count_room) : {Nat.is_le(Nat.add(U32.to_nat((source_index + 1 : U32)),more_count),U32.to_nat(cap)) == True{} : Bool}}
              +named_rest = Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Logical,logicals),1n+List.length(&2,List<&2,U32>,sources),named)
              walk_joined(~Element,~Logical,~original,all,readings,source_index,limits,fill,components,source,sources,model,models,row_when,source_ok,laid,held_empty,good,
                sources_moved(~Element,~Logical,~original,segments,sources,shifts,clamps,logicals,models,all,readings,(source_index + 1 : U32),limits,target,fill,components,separate,
                  facts_rest,placed_rest,named_rest,after_logicals,after_rows,after_room))
            case Moved.NoModels{}:
              Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,Moved.NoModels{},Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,Moved.NoModels{},Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>,placed)
        case Nil{}:
          Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>,Digits.zero_successor(List.length(&2,List<&2,U32>,sources),named))
    case Nil{} Nil{} Nil{} Nil{}:
      Held.Both{Unit{},Held.Both{Unit{},Unit{}}}
    case Con{+own,+segments} Nil{} +shifts +clamps:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Nil{}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n))>>,facts)
    case Con{+own,+segments} Con{+source,+sources} Nil{} +clamps:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>,facts)
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Nil{}:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>,facts)
    case Nil{} Con{+source,+sources} +shifts +clamps:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Con{source,sources}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Con{source,sources}),source_index,models,Components.gones_upto(Law.placing_origins(Con{source,sources}),components,source_index,0n))>>,facts)
    case Nil{} Nil{} Con{+shift,+shifts} +clamps:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Nil{}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n))>>,facts)
    case Nil{} Nil{} Nil{} Con{+clamped,+clamps}:
      Empty.absurd(Held.Both<Components.OriginsOk(limits,Law.placing_origins(Nil{}),readings,source_index,Tasks.separable(readings)),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,fill,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,all),Law.placing_origins(Nil{}),source_index,models,Components.gones_upto(Law.placing_origins(Nil{}),components,source_index,0n))>>,facts)

# ---- The placed tensor ----

# What a placing writes, then a map of every piece: what the map keeps of
# what the placing holds.
def placed_walked(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),~step: @+context: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: PiecesMap.Rule(~Element,~Later,~after,~step),~Outer: @+later: Later -> @+placings: Gathered.Placings<Logical,Element> -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~lift: @+later: Later -> @+placings: Gathered.Placings<Logical,Element> -> @+first: U32 -> @+rows: U32 -> @+inner: Buffer.Content<Element> -> @+holds: Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),placings,first,rows,inner) -> Outer(later,placings,first,rows,step(later,first,rows,inner)),+later: Later,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+fill: Element,+workers: U32,+models: Moved.Models<Element>,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: Law.PlacingFacts(limits,target,segments,sources,shifts,clamps),+walk: Held.Both<Components.OriginsOk(limits,Law.placing_origins(sources),Law.placing_rows(target,sources,shifts,clamps),0,Tasks.separable(Law.placing_rows(target,sources,shifts,clamps))),Held.Both<Moved.SourcesStatic(~Element,~Law.HeldAt(~Element,~Logical,~original,logicals),Law.placing_origins(sources),0,fill,models,Components.gones_upto(Law.placing_origins(sources),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),0,0n)),Moved.SourcesGood(~Element,~Law.HeldAt(~Element,~Logical,~original,logicals),Law.placing_origins(sources),0,models,Components.gones_upto(Law.placing_origins(sources),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),0,0n))>>) ->
  Placed.PlacedModel<Element,Outer(later,Gathered.Placings{logicals,segments,sources,target,fill}),Pieces.ranges_of_grid(Tensors.grid_of(target)),Tensors.moved_with(~Element,~Tensors.Then<Tensors.Placing,Later>,~Tensors.then_gather(~Element,~Tensors.Placing,~Later,~Tensors.placing_gather(~Element),~after),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),Tensors.Then{Tensors.Placing{Tensors.grid_of(target),target,Tensors.origin_shapes(Law.placing_origins(sources)),segments},later},Tensors.move_rows(),0.0,Moved.pack_sources(~Element,models),Law.placing_origins(sources),target,fill,workers)>:
  Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+sources_facts}}}} = facts
  Held.Both{+origins_ok,Held.Both{+static,+good}} = walk
  +readings = Law.placing_rows(target,sources,shifts,clamps)
  +context = {Tensors.Then{Tensors.Placing{Tensors.grid_of(target),target,Tensors.origin_shapes(Law.placing_origins(sources)),segments},later} : Tensors.Then<Tensors.Placing,Later>}
  +same_context = Equal.cong(List<&2,List<&2,U32>>,Tensors.Placing,shapes => Tensors.Placing{Tensors.grid_of(target),target,shapes,segments},Tensors.origin_shapes(Law.placing_origins(sources)),sources,
    shapes_of(sources))
  Components.moved_with_model(~Element,~Tensors.Then<Tensors.Placing,Later>,~Tensors.then_gather(~Element,~Tensors.Placing,~Later,~Tensors.placing_gather(~Element),~after),
    ~Then.then_model(~Element,~Tensors.Placing,~Later,~after,~Gathered.placing_model(~Element)),~Law.HeldAt(~Element,~Logical,~original,logicals),
    ~Outer(later,Gathered.Placings{logicals,segments,sources,target,fill}),~Then.ThenAgrees(~Element,~Tensors.Placing,~Later,~Law.PlacingAgrees(~Element,limits,target,segments,sources,shifts,clamps,fill),later),
    ~Then.then_law(~Element,~Tensors.Placing,~Later,~Tensors.placing_gather(~Element),~Gathered.placing_model(~Element),~after,~step,~rule,~Law.HeldAt(~Element,~Logical,~original,logicals),
      ~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,fill}),~(later => first => rows => buffer => Outer(later,Gathered.Placings{logicals,segments,sources,target,fill},first,rows,buffer)),~Law.PlacingAgrees(~Element,limits,target,segments,sources,shifts,clamps,fill),
      ~Law.placing_law(~Element,~Logical,~original,logicals,limits,target,segments,sources,shifts,clamps,fill,named),
      ~(later => first => rows => inner => holds => lift(later,Gathered.Placings{logicals,segments,sources,target,fill},first,rows,inner,holds)),later),
    Tasks.separable(readings),limits,readings,Tasks.target_rank(readings),Law.placing_origins(sources),target,fill,context,
    Held.Both{{==},Held.Both{same_context,Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},Held.Both{{==},{Held.Both{Held.Both{room,Held.Both{nonempty,rank_room}},Held.Both{components_ok,Held.Both{lists_positive,Held.Both{limits_at,sources_facts}}}} : Law.PlacingFacts(limits,target,segments,sources,shifts,clamps)}}}}}}}},
    Tensors.move_rows(),0.0,workers,models,components_ok,{==},origins_ok,lists_positive,{Held.Both{room,Held.Both{nonempty,rank_room}} : Components.ShapeOk(limits,target)},limits_at,static,good)

# The pieces a placing writes from sources laid as the canonical pieces of
# their shapes, then a map of every piece, hold what the map keeps of what
# the placing holds.
def placing_placed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),~step: @+context: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: PiecesMap.Rule(~Element,~Later,~after,~step),~Outer: @+later: Later -> @+placings: Gathered.Placings<Logical,Element> -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~lift: @+later: Later -> @+placings: Gathered.Placings<Logical,Element> -> @+first: U32 -> @+rows: U32 -> @+inner: Buffer.Content<Element> -> @+holds: Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),placings,first,rows,inner) -> Outer(later,placings,first,rows,step(later,first,rows,inner)),+later: Later,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+fill: Element,+workers: U32,+models: Moved.Models<Element>,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: Law.PlacingFacts(limits,target,segments,sources,shifts,clamps),+placed: SourcesPlaced(~Element,~Logical,~original,logicals,sources,models)) ->
  Placed.PlacedModel<Element,Outer(later,Gathered.Placings{logicals,segments,sources,target,fill}),Pieces.ranges_of_grid(Tensors.grid_of(target)),Tensors.moved_with(~Element,~Tensors.Then<Tensors.Placing,Later>,~Tensors.then_gather(~Element,~Tensors.Placing,~Later,~Tensors.placing_gather(~Element),~after),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),Tensors.Then{Tensors.Placing{Tensors.grid_of(target),target,Tensors.origin_shapes(Law.placing_origins(sources)),segments},later},Tensors.move_rows(),0.0,Moved.pack_sources(~Element,models),Law.placing_origins(sources),target,fill,workers)>:
  Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+sources_facts}}}} = facts
  +readings = Law.placing_rows(target,sources,shifts,clamps)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank_cap = Order.transitive(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits)),U32.to_nat(cap),rank_room,
    Bounds.as_nat_le(Machine.capacity(limits),cap,Limits.capacity_within_half(limits)))
  placed_walked(~Element,~Logical,~original,~Later,~after,~step,~rule,~Outer,~lift,later,logicals,limits,target,segments,sources,shifts,clamps,fill,workers,models,named,
    {Held.Both{Held.Both{room,Held.Both{nonempty,rank_room}},Held.Both{components_ok,Held.Both{lists_positive,Held.Both{limits_at,sources_facts}}}} : Law.PlacingFacts(limits,target,segments,sources,shifts,clamps)},
    sources_moved(~Element,~Logical,~original,segments,sources,shifts,clamps,logicals,models,logicals,readings,0,limits,target,fill,Tasks.components_of(readings),
      Law.rows_separable(target,sources,shifts,clamps,cap,cap_ok,rank_cap),sources_facts,placed,named,{==},{==},Law.rows_room_of(readings,Tasks.target_rank(readings),cap,components_ok)))

# Pieces kept as the gather wrote them hold what the gather holds of them.
def kept_placed(~Element: Data,~Inner: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+layout: Pieces.Ranges,-pieces: Pieces.Pieces(~Element),
  +placed: Placed.PlacedModel<Element,Then.Kept(~Element,~Inner,Unit{}),layout,pieces>) -> Placed.PlacedModel<Element,Inner,layout,pieces>:
  Placed.PlacedModel{+model,+exact,+laid,+holds} = placed
  Placed.PlacedModel{model,exact,laid,PiecesModel.content_map(~Element,~Then.Kept(~Element,~Inner,Unit{}),~Inner,~Then.kept_inner(~Element,~Inner),PiecesModel.content(~Element,model),holds)}

# Sources of the contract placed in a target of counted cells give a tensor
# of the contract that holds the values the sources give its cells from the
# fill.
def placed_contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logicals: List<&2,Logical>,+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+fill: Element,+workers: U32,+models: Moved.Models<Element>,+named: {List.length(&2,Logical,logicals) == List.length(&2,List<&2,U32>,sources) : Nat},+facts: Law.PlacingFacts(limits,target,segments,sources,shifts,clamps),+placed: SourcesPlaced(~Element,~Logical,~original,logicals,sources,models),+count: U32,
  +counted: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>}) ->
  Global.Contract(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,fill},
    Tensors.Tensor{Tensors.placed_from(~Element,Law.placing_rows(target,sources,shifts,clamps),segments,Moved.pack_sources(~Element,models),Law.placing_origins(sources),target,fill,workers),target,fill,Tensors.Usable{},workers}):
  Held.Both{Held.Both{+room,Held.Both{+nonempty,+rank_room}},Held.Both{+components_ok,Held.Both{+lists_positive,Held.Both{+limits_at,+sources_facts}}}} = facts
  GridRanges.grid_contract(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,fill},limits,target,fill,workers,Tensors.moved_with(~Element,~Tensors.Then<Tensors.Placing,Unit>,~Tensors.then_gather(~Element,~Tensors.Placing,~Unit,~Tensors.placing_gather(~Element),~Tensors.unchanged(~Element)),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),Tensors.Then{Tensors.Placing{Tensors.grid_of(target),target,Tensors.origin_shapes(Law.placing_origins(sources)),segments},Unit{}},Tensors.move_rows(),0.0,Moved.pack_sources(~Element,models),Law.placing_origins(sources),target,fill,workers),room,nonempty,1n,
    GridRanges.Multiple{GridRanges.last_extent(target),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(target),1n),GridRanges.last_extent(target),Order.times_one(GridRanges.last_extent(target)))},
    Global.Counted{count,counted},
    kept_placed(~Element,~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),Gathered.Placings{logicals,segments,sources,target,fill}),Pieces.ranges_of_grid(Tensors.grid_of(target)),Tensors.moved_with(~Element,~Tensors.Then<Tensors.Placing,Unit>,~Tensors.then_gather(~Element,~Tensors.Placing,~Unit,~Tensors.placing_gather(~Element),~Tensors.unchanged(~Element)),Tasks.components_of(Law.placing_rows(target,sources,shifts,clamps)),Tensors.Then{Tensors.Placing{Tensors.grid_of(target),target,Tensors.origin_shapes(Law.placing_origins(sources)),segments},Unit{}},Tensors.move_rows(),0.0,Moved.pack_sources(~Element,models),Law.placing_origins(sources),target,fill,workers),
      placing_placed(~Element,~Logical,~original,~Unit,~Tensors.unchanged(~Element),~Then.kept_step(~Element),~Then.kept_rule(~Element),~(later => placings => Then.Kept(~Element,~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),placings),later)),~(later => placings => first => rows => inner => holds => Then.kept_lift(~Element,~Placed.PieceHolds(~Element,~Gathered.Placings<Logical,Element>,~Gathered.placing_value(~Element,~Logical,~original),placings),later,first,rows,inner,holds)),Unit{},logicals,limits,target,segments,sources,shifts,clamps,fill,workers,models,named,{Held.Both{Held.Both{room,Held.Both{nonempty,rank_room}},Held.Both{components_ok,Held.Both{lists_positive,Held.Both{limits_at,sources_facts}}}} : Law.PlacingFacts(limits,target,segments,sources,shifts,clamps)},placed)))
