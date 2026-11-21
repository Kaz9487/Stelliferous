# The parts that place one source in a target piece. On every axis each
# segment that reaches the piece gives a box axis, and the parts are the boxes
# of every choice of one such axis per axis, outer axes first. A target cell
# ends with the value of the source cell that the last segment covering its
# coordinate on every axis reads, or keeps its value when an axis has none.
import Base
import ../tensor_move.bend as Move
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./tensor_move_part.bend as Part
import ./tensor_move_parts.bend as MoveParts
import ./tensor_box_parts.bend as BoxParts
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./word_arithmetic.bend as WordArithmetic
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm
import ./tensor_grid_pieces.bend as GridLaws

# ---- The parts as boxes ----

# The box axes of the segments of one axis inside a piece: the cut of every
# segment that reaches the piece, with its start in the piece.
def axis_boxes(segments: List<&2,Tasks.Segment>,+low: U32,+extent: U32,+source_stride: U32,+piece_stride: U32) -> List<&2,BoxParts.BoxAxis>:
  match segments:
    case Nil{}: []
    case Con{Tasks.Segment{+first,+count,+source,+step},rest}:
      +start = U32.max(first,low)
      +end = U32.min((first + count : U32),(low + extent : U32))
      +later = axis_boxes(rest,low,extent,source_stride,piece_stride)
      Bool.pick(List<&2,BoxParts.BoxAxis>,U32.is_lt(start,end),
        BoxParts.BoxAxis{Tasks.Cut{(end - start : U32),(step * source_stride : U32),((source + (start - first) * step) * source_stride : U32),piece_stride,((start - low) * piece_stride : U32)},
          U32.to_nat((start - low : U32)),extent} <> later,later)

def box_cuts(list: List<&2,BoxParts.BoxAxis>) -> List<&2,Tasks.Cut>:
  match list:
    case Nil{}: []
    case Con{BoxParts.BoxAxis{+cut,start,extent},rest}: cut <> box_cuts(rest)

def picked_cuts(found: Bool,+cut: Tasks.Cut,+start: Nat,+extent: U32,+later: List<&2,BoxParts.BoxAxis>) ->
  {box_cuts(Bool.pick(List<&2,BoxParts.BoxAxis>,found,BoxParts.BoxAxis{cut,start,extent} <> later,later)) ==
    Bool.pick(List<&2,Tasks.Cut>,found,cut <> box_cuts(later),box_cuts(later)) : List<&2,Tasks.Cut>}:
  match found:
    case True{}: {==}
    case False{}: {==}

# The box axes of an axis hold the cuts of the axis.
def axis_cuts_boxes(segments: List<&2,Tasks.Segment>,+low: U32,+extent: U32,+source_stride: U32,+piece_stride: U32) ->
  {box_cuts(axis_boxes(segments,low,extent,source_stride,piece_stride)) == Tasks.axis_cuts(segments,low,extent,source_stride,piece_stride) : List<&2,Tasks.Cut>}:
  match segments:
    case Nil{}: {==}
    case Con{Tasks.Segment{+first,+count,+source,+step},+rest}:
      +start = U32.max(first,low)
      +end = U32.min((first + count : U32),(low + extent : U32))
      +found = U32.is_lt(start,end)
      +cut = {Tasks.Cut{(end - start : U32),(step * source_stride : U32),((source + (start - first) * step) * source_stride : U32),piece_stride,((start - low) * piece_stride : U32)} : Tasks.Cut}
      +later = axis_boxes(rest,low,extent,source_stride,piece_stride)
      +cuts = Tasks.axis_cuts(rest,low,extent,source_stride,piece_stride)
      Equal.trans(List<&2,Tasks.Cut>,box_cuts(Bool.pick(List<&2,BoxParts.BoxAxis>,found,BoxParts.BoxAxis{cut,U32.to_nat((start - low : U32)),extent} <> later,later)),
        Bool.pick(List<&2,Tasks.Cut>,found,cut <> box_cuts(later),box_cuts(later)),Bool.pick(List<&2,Tasks.Cut>,found,cut <> cuts,cuts),
        picked_cuts(found,cut,U32.to_nat((start - low : U32)),extent,later),
        Equal.cong(List<&2,Tasks.Cut>,List<&2,Tasks.Cut>,tail => Bool.pick(List<&2,Tasks.Cut>,found,cut <> tail,tail),box_cuts(later),cuts,
          axis_cuts_boxes(rest,low,extent,source_stride,piece_stride)))

# A box of a piece being built, its axes innermost first, with the first
# source and target cells of its cuts so far.
type Prefix is Data:
  Prefix{boxes: List<&2,BoxParts.BoxAxis>,source: U32,target: U32}

def prefix_extended(boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+target: U32,+later: List<&2,Prefix>) -> List<&2,Prefix>:
  match boxes:
    case Nil{}: later
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_add,+target_stride,+target_add},+start,+extent},rest}:
      Prefix{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_add,target_stride,target_add},start,extent} <> inner,(source + source_add : U32),(target + target_add : U32)} <>
        prefix_extended(rest,inner,source,target,later)

def prefixes_extended(prefixes: List<&2,Prefix>,+boxes: List<&2,BoxParts.BoxAxis>) -> List<&2,Prefix>:
  match prefixes:
    case Nil{}: []
    case Con{Prefix{+inner,+source,+target},rest}: prefix_extended(boxes,inner,source,target,prefixes_extended(rest,boxes))

def placed_prefixes(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  +prefixes: List<&2,Prefix>) -> List<&2,Prefix>:
  match segments coordinates extents source_strides piece_strides:
    case Con{own,segments} Con{+low,coordinates} Con{+extent,extents} Con{+source_stride,source_strides} Con{+piece_stride,piece_strides}:
      placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,prefixes_extended(prefixes,axis_boxes(own,low,extent,source_stride,piece_stride)))
    case segments coordinates extents source_strides piece_strides: prefixes

def prefix_partials(list: List<&2,Prefix>) -> List<&2,Tasks.Partial>:
  match list:
    case Nil{}: []
    case Con{Prefix{+boxes,+source,+target},rest}: Tasks.Partial{BoxParts.box_axes(boxes),source,target} <> prefix_partials(rest)

def extended_box(boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+target: U32,+later: List<&2,Prefix>) ->
  {prefix_partials(prefix_extended(boxes,inner,source,target,later)) == Tasks.extended_partial(box_cuts(boxes),BoxParts.box_axes(inner),source,target,prefix_partials(later)) :
    List<&2,Tasks.Partial>}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_add,+target_stride,+target_add},+start,+extent},+rest}:
      Equal.cong(List<&2,Tasks.Partial>,List<&2,Tasks.Partial>,
        tail => Tasks.Partial{Move.Axis{count,source_stride,False{},target_stride} <> BoxParts.box_axes(inner),(source + source_add : U32),(target + target_add : U32)} <> tail,
        prefix_partials(prefix_extended(rest,inner,source,target,later)),Tasks.extended_partial(box_cuts(rest),BoxParts.box_axes(inner),source,target,prefix_partials(later)),
        extended_box(rest,inner,source,target,later))

def extended_boxes(prefixes: List<&2,Prefix>,+boxes: List<&2,BoxParts.BoxAxis>) ->
  {prefix_partials(prefixes_extended(prefixes,boxes)) == Tasks.extended(prefix_partials(prefixes),box_cuts(boxes)) : List<&2,Tasks.Partial>}:
  match prefixes:
    case Nil{}: {==}
    case Con{Prefix{+inner,+source,+target},+rest}:
      Equal.trans(List<&2,Tasks.Partial>,prefix_partials(prefix_extended(boxes,inner,source,target,prefixes_extended(rest,boxes))),
        Tasks.extended_partial(box_cuts(boxes),BoxParts.box_axes(inner),source,target,prefix_partials(prefixes_extended(rest,boxes))),
        Tasks.extended_partial(box_cuts(boxes),BoxParts.box_axes(inner),source,target,Tasks.extended(prefix_partials(rest),box_cuts(boxes))),
        extended_box(boxes,inner,source,target,prefixes_extended(rest,boxes)),
        Equal.cong(List<&2,Tasks.Partial>,List<&2,Tasks.Partial>,later => Tasks.extended_partial(box_cuts(boxes),BoxParts.box_axes(inner),source,target,later),
          prefix_partials(prefixes_extended(rest,boxes)),Tasks.extended(prefix_partials(rest),box_cuts(boxes)),extended_boxes(rest,boxes)))

# The prefixes give the partial boxes of the placement.
def placed_boxes(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  +prefixes: List<&2,Prefix>) ->
  {prefix_partials(placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,prefixes)) ==
    Tasks.placed_axes(segments,coordinates,extents,source_strides,piece_strides,prefix_partials(prefixes)) : List<&2,Tasks.Partial>}:
  match segments:
    case Nil{}: {==}
    case Con{+own,+segments}:
      match coordinates:
        case Nil{}: {==}
        case Con{+low,+coordinates}:
          match extents:
            case Nil{}: {==}
            case Con{+extent,+extents}:
              match source_strides:
                case Nil{}: {==}
                case Con{+source_stride,+source_strides}:
                  match piece_strides:
                    case Nil{}: {==}
                    case Con{+piece_stride,+piece_strides}:
                      +boxes = axis_boxes(own,low,extent,source_stride,piece_stride)
                      +cuts = Tasks.axis_cuts(own,low,extent,source_stride,piece_stride)
                      +next = prefixes_extended(prefixes,boxes)
                      Equal.trans(List<&2,Tasks.Partial>,prefix_partials(placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,next)),
                        Tasks.placed_axes(segments,coordinates,extents,source_strides,piece_strides,prefix_partials(next)),
                        Tasks.placed_axes(segments,coordinates,extents,source_strides,piece_strides,Tasks.extended(prefix_partials(prefixes),cuts)),
                        placed_boxes(segments,coordinates,extents,source_strides,piece_strides,next),
                        Equal.cong(List<&2,Tasks.Partial>,List<&2,Tasks.Partial>,partials => Tasks.placed_axes(segments,coordinates,extents,source_strides,piece_strides,partials),
                          prefix_partials(next),Tasks.extended(prefix_partials(prefixes),cuts),
                          Equal.trans(List<&2,Tasks.Partial>,prefix_partials(next),Tasks.extended(prefix_partials(prefixes),box_cuts(boxes)),Tasks.extended(prefix_partials(prefixes),cuts),
                            extended_boxes(prefixes,boxes),
                            Equal.cong(List<&2,Tasks.Cut>,List<&2,Tasks.Partial>,list => Tasks.extended(prefix_partials(prefixes),list),box_cuts(boxes),cuts,
                              axis_cuts_boxes(own,low,extent,source_stride,piece_stride)))))

def reversed_boxes(boxes: List<&2,BoxParts.BoxAxis>,+done: List<&2,BoxParts.BoxAxis>) -> List<&2,BoxParts.BoxAxis>:
  match boxes:
    case Nil{}: done
    case Con{+box,rest}: reversed_boxes(rest,box <> done)

def reversed_box_axes(boxes: List<&2,BoxParts.BoxAxis>,+done: List<&2,BoxParts.BoxAxis>) ->
  {Tasks.reversed_axes(BoxParts.box_axes(boxes),BoxParts.box_axes(done)) == BoxParts.box_axes(reversed_boxes(boxes,done)) : List<&2,Move.Axis>}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_add,+target_stride,+target_add},+start,+extent},+rest}:
      reversed_box_axes(rest,BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_add,target_stride,target_add},start,extent} <> done)

# The part of a finished prefix: its box with the outer axis first.
def prefix_parts(list: List<&2,Prefix>) -> List<&2,Move.Part>:
  match list:
    case Nil{}: []
    case Con{Prefix{+boxes,+source,+target},rest}: Move.Part{BoxParts.box_axes(reversed_boxes(boxes,[])),source,target} <> prefix_parts(rest)

def parts_of_prefixes(list: List<&2,Prefix>) -> {Tasks.parts_of(prefix_partials(list)) == prefix_parts(list) : List<&2,Move.Part>}:
  match list:
    case Nil{}: {==}
    case Con{Prefix{+boxes,+source,+target},+rest}:
      %Equal.sym(List<&2,Move.Axis>,Tasks.reversed_axes(BoxParts.box_axes(boxes),[]),BoxParts.box_axes(reversed_boxes(boxes,[])),reversed_box_axes(boxes,[])) : {Move.Part{_,source,target} <> Tasks.parts_of(prefix_partials(rest)) == prefix_parts(Con{Prefix{boxes,source,target},rest}) : List<&2,Move.Part>}
      Equal.cong(List<&2,Move.Part>,List<&2,Move.Part>,tail => Move.Part{BoxParts.box_axes(reversed_boxes(boxes,[])),source,target} <> tail,
        Tasks.parts_of(prefix_partials(rest)),prefix_parts(rest),parts_of_prefixes(rest))

# The parts that place a source in a target piece are the parts of the
# prefixes built axis after axis from the empty one.
def placements_prefixes(+segments: List<&2,List<&2,Tasks.Segment>>,+source: List<&2,U32>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,+count: U32) ->
  {Tasks.placements(segments,source,grid,cells,first,count) ==
    prefix_parts(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),
      [Prefix{[],0,0}])) : List<&2,Move.Part>}:
  +extents = Grid.box_of(grid,cells,count)
  +prefixes = placed_prefixes(segments,Grid.coordinates(cells,first),extents,Tasks.strides_of(source),Tasks.strides_of(extents),[Prefix{[],0,0}])
  Equal.trans(List<&2,Move.Part>,Tasks.parts_of(Tasks.placed_axes(segments,Grid.coordinates(cells,first),extents,Tasks.strides_of(source),Tasks.strides_of(extents),[Tasks.Partial{[],0,0}])),
    Tasks.parts_of(prefix_partials(prefixes)),prefix_parts(prefixes),
    Equal.cong(List<&2,Tasks.Partial>,List<&2,Move.Part>,Tasks.parts_of,
      Tasks.placed_axes(segments,Grid.coordinates(cells,first),extents,Tasks.strides_of(source),Tasks.strides_of(extents),[Tasks.Partial{[],0,0}]),prefix_partials(prefixes),
      Equal.sym(List<&2,Tasks.Partial>,prefix_partials(prefixes),
        Tasks.placed_axes(segments,Grid.coordinates(cells,first),extents,Tasks.strides_of(source),Tasks.strides_of(extents),[Tasks.Partial{[],0,0}]),
        placed_boxes(segments,Grid.coordinates(cells,first),extents,Tasks.strides_of(source),Tasks.strides_of(extents),[Prefix{[],0,0}]))),
    parts_of_prefixes(prefixes))

# ---- The boxes of the finished prefixes ----

def reversed_extents(list: List<&2,U32>,+done: List<&2,U32>) -> List<&2,U32>:
  match list:
    case Nil{}: done
    case Con{+extent,rest}: reversed_extents(rest,extent <> done)

# Every box axis of a prefix, innermost first, against the extents of the
# axes done, innermost first, with the extents of the axes inside it: it has
# the extent of its axis, steps and starts with the piece stride, is not empty
# and ends inside the piece.
def PrefixIn(boxes: List<&2,BoxParts.BoxAxis>,done: List<&2,U32>,+rest: List<&2,U32>) -> Data:
  match boxes done:
    case Nil{} Nil{}: Unit
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,source_stride,source_offset,+target_stride,+target_offset},+start,+own},more} Con{+extent,done}:
      Held.Both<{own == extent : U32},Held.Both<{target_stride == Grid.product(rest) : U32},Held.Both<{target_offset == (U32.from_nat(start) * Grid.product(rest) : U32) : U32},Held.Both<{Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},Held.Both<{Nat.is_le(Nat.add(start,U32.to_nat(count)),U32.to_nat(own)) == True{} : Bool},PrefixIn(more,done,extent <> rest)>>>>>
    case Nil{} Con{extent,done}: Empty
    case Con{box,more} Nil{}: Empty

# A finished prefix is a box inside the piece.
def prefix_box_in(boxes: List<&2,BoxParts.BoxAxis>,done: List<&2,U32>,+outer: List<&2,BoxParts.BoxAxis>,+inside: PrefixIn(boxes,done,BoxParts.box_extents(outer)),
  +held: BoxParts.BoxIn(outer)) -> BoxParts.BoxIn(reversed_boxes(boxes,outer)):
  match boxes done:
    case Nil{} Nil{}: held
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more} Con{+extent,+done}:
      Held.Both{+same,Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+inner}}}}} = inside
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      prefix_box_in(more,done,box <> outer,
        {%Equal.sym(U32,own,extent,same) : PrefixIn(more,done,_ <> BoxParts.box_extents(outer)) inner : PrefixIn(more,done,own <> BoxParts.box_extents(outer))},
        Held.Both{stride,Held.Both{offset,Held.Both{positive,Held.Both{fits,held}}}})
    case Nil{} Con{+extent,+done}: Empty.absurd(BoxParts.BoxIn(outer),inside)
    case Con{+box,+more} Nil{}: Empty.absurd(BoxParts.BoxIn(reversed_boxes(more,box <> outer)),inside)

# A finished prefix has the extents of the axes done.
def prefix_box_extents(boxes: List<&2,BoxParts.BoxAxis>,done: List<&2,U32>,+outer: List<&2,BoxParts.BoxAxis>,+inside: PrefixIn(boxes,done,BoxParts.box_extents(outer))) ->
  {BoxParts.box_extents(reversed_boxes(boxes,outer)) == reversed_extents(done,BoxParts.box_extents(outer)) : List<&2,U32>}:
  match boxes done:
    case Nil{} Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more} Con{+extent,+done}:
      Held.Both{+same,Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+inner}}}}} = inside
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Equal.trans(List<&2,U32>,BoxParts.box_extents(reversed_boxes(more,box <> outer)),reversed_extents(done,own <> BoxParts.box_extents(outer)),
        reversed_extents(done,extent <> BoxParts.box_extents(outer)),
        prefix_box_extents(more,done,box <> outer,
          {%Equal.sym(U32,own,extent,same) : PrefixIn(more,done,_ <> BoxParts.box_extents(outer)) inner : PrefixIn(more,done,own <> BoxParts.box_extents(outer))}),
        Equal.cong(U32,List<&2,U32>,first => reversed_extents(done,first <> BoxParts.box_extents(outer)),own,extent,same))
    case Nil{} Con{+extent,+done}: Empty.absurd({BoxParts.box_extents(outer) == reversed_extents(Con{extent,done},BoxParts.box_extents(outer)) : List<&2,U32>},inside)
    case Con{+box,+more} Nil{}: Empty.absurd({BoxParts.box_extents(reversed_boxes(more,box <> outer)) == BoxParts.box_extents(outer) : List<&2,U32>},inside)

def target_reversed(boxes: List<&2,BoxParts.BoxAxis>,+outer: List<&2,BoxParts.BoxAxis>,+first: U32) ->
  {BoxParts.box_target(reversed_boxes(boxes,outer),first) == BoxParts.box_target(outer,BoxParts.box_target(reversed_boxes(boxes,[]),first)) : U32}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Equal.trans(U32,BoxParts.box_target(reversed_boxes(more,box <> outer),first),BoxParts.box_target(box <> outer,BoxParts.box_target(reversed_boxes(more,[]),first)),
        BoxParts.box_target(outer,BoxParts.box_target(reversed_boxes(more,[box]),first)),
        target_reversed(more,box <> outer,first),
        Equal.cong(U32,U32,value => BoxParts.box_target(outer,value),(BoxParts.box_target(reversed_boxes(more,[]),first) + target_offset : U32),
          BoxParts.box_target(reversed_boxes(more,[box]),first),
          Equal.sym(U32,BoxParts.box_target(reversed_boxes(more,[box]),first),BoxParts.box_target([box],BoxParts.box_target(reversed_boxes(more,[]),first)),
            target_reversed(more,[box],first))))

# Every prefix of a list is in the piece and writes from the first target cell
# of its box.
def PrefixesIn(list: List<&2,Prefix>,+done: List<&2,U32>,+rest: List<&2,U32>) -> Data:
  match list:
    case Nil{}: Unit
    case Con{Prefix{+boxes,+source,+target},more}:
      Held.Both<PrefixIn(boxes,done,rest),Held.Both<{target == BoxParts.box_target(reversed_boxes(boxes,[]),0) : U32},PrefixesIn(more,done,rest)>>

# Every box axis of an axis is in the piece, with the extents of the axes
# inside it.
def AxisBoxesIn(boxes: List<&2,BoxParts.BoxAxis>,+extent: U32,+rest: List<&2,U32>) -> Data:
  match boxes:
    case Nil{}: Unit
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,source_stride,source_offset,+target_stride,+target_offset},+start,+own},more}:
      Held.Both<{own == extent : U32},Held.Both<{target_stride == Grid.product(rest) : U32},Held.Both<{target_offset == (U32.from_nat(start) * Grid.product(rest) : U32) : U32},Held.Both<{Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},Held.Both<{Nat.is_le(Nat.add(start,U32.to_nat(count)),U32.to_nat(own)) == True{} : Bool},AxisBoxesIn(more,extent,rest)>>>>>

def prefix_extended_in(boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+target: U32,+later: List<&2,Prefix>,+extent: U32,+done: List<&2,U32>,
  +rest: List<&2,U32>,+axis: AxisBoxesIn(boxes,extent,rest),+held: PrefixIn(inner,done,extent <> rest),+placed: {target == BoxParts.box_target(reversed_boxes(inner,[]),0) : U32},
  +later_in: PrefixesIn(later,extent <> done,rest)) -> PrefixesIn(prefix_extended(boxes,inner,source,target,later),extent <> done,rest):
  match boxes:
    case Nil{}: later_in
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      Held.Both{+same,Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+axis_more}}}}} = axis
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Held.Both{Held.Both{same,Held.Both{stride,Held.Both{offset,Held.Both{positive,Held.Both{fits,held}}}}},
        Held.Both{Equal.trans(U32,(target + target_offset : U32),(BoxParts.box_target(reversed_boxes(inner,[]),0) + target_offset : U32),BoxParts.box_target(reversed_boxes(inner,[box]),0),
            Equal.cong(U32,U32,value => (value + target_offset : U32),target,BoxParts.box_target(reversed_boxes(inner,[]),0),placed),
            Equal.sym(U32,BoxParts.box_target(reversed_boxes(inner,[box]),0),BoxParts.box_target([box],BoxParts.box_target(reversed_boxes(inner,[]),0)),target_reversed(inner,[box],0))),
          prefix_extended_in(more,inner,source,target,later,extent,done,rest,axis_more,held,placed,later_in)}}

def prefixes_extended_in(prefixes: List<&2,Prefix>,+boxes: List<&2,BoxParts.BoxAxis>,+extent: U32,+done: List<&2,U32>,+rest: List<&2,U32>,+axis: AxisBoxesIn(boxes,extent,rest),
  +held: PrefixesIn(prefixes,done,extent <> rest)) -> PrefixesIn(prefixes_extended(prefixes,boxes),extent <> done,rest):
  match prefixes:
    case Nil{}: Unit{}
    case Con{Prefix{+inner,+source,+target},+more}:
      Held.Both{+own,Held.Both{+placed,+held_more}} = held
      prefix_extended_in(boxes,inner,source,target,prefixes_extended(more,boxes),extent,done,rest,axis,own,placed,prefixes_extended_in(more,boxes,extent,done,rest,axis,held_more))

# Every axis has box axes in the piece, and every axis of the piece has
# segments.
def AxesBoxed(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>) -> Data:
  match segments coordinates extents source_strides piece_strides:
    case Con{own,segments} Con{+low,coordinates} Con{+extent,extents} Con{+source_stride,source_strides} Con{+piece_stride,piece_strides}:
      Held.Both<AxisBoxesIn(axis_boxes(own,low,extent,source_stride,piece_stride),extent,extents),AxesBoxed(segments,coordinates,extents,source_strides,piece_strides)>
    case segments coordinates extents source_strides piece_strides: {extents == [] : List<&2,U32>}

def ended_in(+prefixes: List<&2,Prefix>,+done: List<&2,U32>,+extents: List<&2,U32>,+ended: {extents == [] : List<&2,U32>},+held: PrefixesIn(prefixes,done,extents)) ->
  PrefixesIn(prefixes,reversed_extents(extents,done),[]):
  %Equal.sym(List<&2,U32>,extents,[],ended) : PrefixesIn(prefixes,reversed_extents(_,done),[])
  %ended : PrefixesIn(prefixes,done,_) held

# The prefixes of a placement are boxes in the piece.
def prefixes_in(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  +prefixes: List<&2,Prefix>,+done: List<&2,U32>,+axes: AxesBoxed(segments,coordinates,extents,source_strides,piece_strides),+held: PrefixesIn(prefixes,done,extents)) ->
  PrefixesIn(placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,prefixes),reversed_extents(extents,done),[]):
  match segments:
    case Nil{}: ended_in(prefixes,done,extents,axes,held)
    case Con{+own,+segments}:
      match coordinates:
        case Nil{}: ended_in(prefixes,done,extents,axes,held)
        case Con{+low,+coordinates}:
          match extents:
            case Nil{}: ended_in(prefixes,done,[],axes,held)
            case Con{+extent,+extents}:
              match source_strides:
                case Nil{}: ended_in(prefixes,done,extent <> extents,axes,held)
                case Con{+source_stride,+source_strides}:
                  match piece_strides:
                    case Nil{}: ended_in(prefixes,done,extent <> extents,axes,held)
                    case Con{+piece_stride,+piece_strides}:
                      Held.Both{+axis,+axes_more} = axes
                      +boxes = axis_boxes(own,low,extent,source_stride,piece_stride)
                      prefixes_in(segments,coordinates,extents,source_strides,piece_strides,prefixes_extended(prefixes,boxes),extent <> done,axes_more,
                        prefixes_extended_in(prefixes,boxes,extent,done,extents,axis,held))

# ---- The value each target cell takes ----

# Prefix after prefix, the value of the cell query: a box that holds it
# writes the source cell it reads there, others leave it.
def boxes_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Prefix>,+query: U32,+old: Element) -> Element:
  match list:
    case Nil{}: old
    case Con{Prefix{+boxes,+source,+target},rest}:
      boxes_value(~Element,~Context,~value,context,rest,query,Bool.pick(Element,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old))

def prefix_indices(list: List<&2,Prefix>,+query: U32) -> List<&2,Nat>:
  match list:
    case Nil{}: []
    case Con{Prefix{+boxes,source,target},rest}: BoxParts.hit_index(reversed_boxes(boxes,[]),query) <> prefix_indices(rest,query)

def box_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,found: Bool,+boxes: List<&2,BoxParts.BoxAxis>,+source: U32,+target: U32,+rest: List<&2,Prefix>,+query: U32,+old: Element,
  +inside: BoxParts.BoxIn(reversed_boxes(boxes,[])),+placed: {target == BoxParts.box_target(reversed_boxes(boxes,[]),0) : U32},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))),U32.to_nat(cap)) == True{} : Bool},
  +query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))) == True{} : Bool},+holds: {BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))) == found : Bool},
  +hit_rest: MoveParts.PartsAt(~Element,~Context,~value,context,prefix_parts(rest),prefix_indices(rest,query),query,value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),boxes_value(~Element,~Context,~value,context,rest,query,value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)))),
  +miss_rest: MoveParts.PartsAt(~Element,~Context,~value,context,prefix_parts(rest),prefix_indices(rest,query),query,old,boxes_value(~Element,~Context,~value,context,rest,query,old))) ->
  MoveParts.PartsAt(~Element,~Context,~value,context,Con{Move.Part{BoxParts.box_axes(reversed_boxes(boxes,[])),source,target},prefix_parts(rest)},Con{BoxParts.hit_index(reversed_boxes(boxes,[]),query),prefix_indices(rest,query)},query,old,
    boxes_value(~Element,~Context,~value,context,rest,query,Bool.pick(Element,found,value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old))):
  match found:
    case True{}:
      +index_in = BoxParts.hit_below(reversed_boxes(boxes,[]),query,holds)
      MoveParts.Hits{Held.Both{Held.Both{index_in,Held.Both{
          {%Equal.sym(U32,target,BoxParts.box_target(reversed_boxes(boxes,[]),0),placed) : {Part.cell_target(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,_) == query : U32}
            BoxParts.hit_cell(reversed_boxes(boxes,[]),inside,source,cap,cap_ok,piece_room,query,query_in,holds) :
            {Part.cell_target(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target) == query : U32}},
          {%Equal.sym(U32,target,BoxParts.box_target(reversed_boxes(boxes,[]),0),placed) : Part.CellsApart(BoxParts.box_axes(reversed_boxes(boxes,[])),source,_,BoxParts.hit_index(reversed_boxes(boxes,[]),query),0n,Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))
            BoxParts.box_apart(Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))),reversed_boxes(boxes,[]),inside,source,cap,cap_ok,piece_room,BoxParts.hit_index(reversed_boxes(boxes,[]),query),index_in,0n,Order.reflexive(Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))) :
            Part.CellsApart(BoxParts.box_axes(reversed_boxes(boxes,[])),source,target,BoxParts.hit_index(reversed_boxes(boxes,[]),query),0n,Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))}}},hit_rest}}
    case False{}:
      MoveParts.Misses{Held.Both{
        {%Equal.sym(U32,target,BoxParts.box_target(reversed_boxes(boxes,[]),0),placed) : Part.CellsMiss(BoxParts.box_axes(reversed_boxes(boxes,[])),source,_,query,0n,Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))
          BoxParts.box_miss(Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))),reversed_boxes(boxes,[]),inside,source,cap,cap_ok,piece_room,query,query_in,holds,0n,Order.reflexive(Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))) :
          Part.CellsMiss(BoxParts.box_axes(reversed_boxes(boxes,[])),source,target,query,0n,Part.total(BoxParts.box_axes(reversed_boxes(boxes,[]))))},miss_rest}}

# Part after part of finished prefixes, the cell query goes from old to the
# value of the last box that holds it.
def boxes_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Prefix>,+done: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(reversed_extents(done,[]))),U32.to_nat(cap)) == True{} : Bool},+query: U32,+query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(reversed_extents(done,[]))) == True{} : Bool},+old: Element,
  +held: PrefixesIn(list,done,[])) ->
  MoveParts.PartsAt(~Element,~Context,~value,context,prefix_parts(list),prefix_indices(list,query),query,old,boxes_value(~Element,~Context,~value,context,list,query,old)):
  match list:
    case Nil{}: {==}
    case Con{Prefix{+boxes,+source,+target},+rest}:
      Held.Both{+own,Held.Both{+placed,+held_rest}} = held
      +same = prefix_box_extents(boxes,done,[],own)
      box_at(~Element,~Context,~value,context,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),boxes,source,target,rest,query,old,prefix_box_in(boxes,done,[],own,Unit{}),placed,cap,cap_ok,
        {%Equal.sym(List<&2,U32>,BoxParts.box_extents(reversed_boxes(boxes,[])),reversed_extents(done,[]),same) : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool} piece_room : {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))),U32.to_nat(cap)) == True{} : Bool}},
        {%Equal.sym(List<&2,U32>,BoxParts.box_extents(reversed_boxes(boxes,[])),reversed_extents(done,[]),same) : {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(_)) == True{} : Bool} query_in : {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))) == True{} : Bool}},{==},
        boxes_at(~Element,~Context,~value,context,rest,done,cap,cap_ok,piece_room,query,query_in,value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),held_rest),
        boxes_at(~Element,~Context,~value,context,rest,done,cap,cap_ok,piece_room,query,query_in,old,held_rest))

# ---- The value of a target cell, axis after axis ----

def reversed_digits(list: List<&2,Nat>,+done: List<&2,Nat>) -> List<&2,Nat>:
  match list:
    case Nil{}: done
    case Con{+digit,rest}: reversed_digits(rest,digit <> done)

def extents_twice(list: List<&2,U32>,+done: List<&2,U32>) -> {reversed_extents(reversed_extents(list,done),[]) == reversed_extents(done,list) : List<&2,U32>}:
  match list:
    case Nil{}: {==}
    case Con{+extent,+rest}: extents_twice(rest,extent <> done)

def digits_twice(list: List<&2,Nat>,+done: List<&2,Nat>) -> {reversed_digits(reversed_digits(list,done),[]) == reversed_digits(done,list) : List<&2,Nat>}:
  match list:
    case Nil{}: {==}
    case Con{+digit,+rest}: digits_twice(rest,digit <> done)

def below_reversed(extents: List<&2,U32>,digits: List<&2,Nat>,+done: List<&2,U32>,+later: List<&2,Nat>,+below: Coordinates.DigitsBelow(extents,digits),
  +held: Coordinates.DigitsBelow(done,later)) -> Coordinates.DigitsBelow(reversed_extents(extents,done),reversed_digits(digits,later)):
  match extents digits:
    case Nil{} Nil{}: held
    case Con{+extent,+rest} Con{+digit,+more}:
      KernelGemm.Both{+lower,+below_more} = below
      below_reversed(rest,more,extent <> done,digit <> later,below_more,KernelGemm.Both{lower,held})
    case Nil{} Con{+digit,+more}: Empty.absurd(Coordinates.DigitsBelow(done,reversed_digits(more,digit <> later)),below)
    case Con{+extent,+rest} Nil{}: Empty.absurd(Coordinates.DigitsBelow(reversed_extents(rest,extent <> done),later),below)

# A prefix holds the digits of the axes done, innermost first, when every box
# axis holds its digit.
def prefix_holds(boxes: List<&2,BoxParts.BoxAxis>,digits: List<&2,Nat>) -> Bool:
  match boxes digits:
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,source_stride,source_offset,target_stride,target_offset},+start,extent},rest} Con{+digit,more}:
      Nat.is_le(start,digit) && (Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && prefix_holds(rest,more))
    case Nil{} Nil{}: True{}
    case boxes digits: False{}

# The source cells a prefix steps to reach the digits of the axes done.
def prefix_offset(boxes: List<&2,BoxParts.BoxAxis>,digits: List<&2,Nat>) -> U32:
  match boxes digits:
    case Con{BoxParts.BoxAxis{Tasks.Cut{count,+source_stride,source_offset,target_stride,target_offset},+start,extent},rest} Con{+digit,more}:
      (U32.from_nat(Nat.sub(digit,start)) * source_stride + prefix_offset(rest,more) : U32)
    case boxes digits: 0

def and_moved(first: Bool,second: Bool,held: Bool,+outer: Bool) -> {held && (first && (second && outer)) == (first && (second && held)) && outer : Bool}:
  match first second held:
    case True{} True{} True{}: {==}
    case True{} True{} False{}: {==}
    case True{} False{} True{}: {==}
    case True{} False{} False{}: {==}
    case False{} True{} True{}: {==}
    case False{} True{} False{}: {==}
    case False{} False{} True{}: {==}
    case False{} False{} False{}: {==}

def and_true(value: Bool) -> {value && True{} == value : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

# A finished prefix holds its digits, outer axis first, when it holds them
# innermost first.
def holds_reversed(boxes: List<&2,BoxParts.BoxAxis>,done: List<&2,U32>,digits: List<&2,Nat>,+outer: List<&2,BoxParts.BoxAxis>,+later: List<&2,Nat>,+rest: List<&2,U32>,
  +inside: PrefixIn(boxes,done,rest),+below: Coordinates.DigitsBelow(done,digits)) ->
  {BoxParts.box_holds(reversed_boxes(boxes,outer),reversed_digits(digits,later)) == prefix_holds(boxes,digits) && BoxParts.box_holds(outer,later) : Bool}:
  match boxes done digits:
    case Nil{} Nil{} Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more} Con{+extent,+done} Con{+digit,+digits}:
      Held.Both{+same,Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+inner}}}}} = inside
      KernelGemm.Both{+lower,+below_more} = below
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Equal.trans(Bool,BoxParts.box_holds(reversed_boxes(more,box <> outer),reversed_digits(digits,digit <> later)),
        prefix_holds(more,digits) && BoxParts.box_holds(box <> outer,digit <> later),
        (Nat.is_le(start,digit) && (Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && prefix_holds(more,digits))) && BoxParts.box_holds(outer,later),
        holds_reversed(more,done,digits,box <> outer,digit <> later,extent <> rest,inner,below_more),
        and_moved(Nat.is_le(start,digit),Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))),prefix_holds(more,digits),BoxParts.box_holds(outer,later)))
    case Nil{} Nil{} Con{+digit,+digits}:
      Empty.absurd({BoxParts.box_holds(outer,reversed_digits(digits,digit <> later)) == prefix_holds(Nil{},Con{digit,digits}) && BoxParts.box_holds(outer,later) : Bool},below)
    case Nil{} Con{+extent,+done} +digits:
      Empty.absurd({BoxParts.box_holds(outer,reversed_digits(digits,later)) == prefix_holds(Nil{},digits) && BoxParts.box_holds(outer,later) : Bool},inside)
    case Con{+box,+more} Nil{} +digits:
      Empty.absurd({BoxParts.box_holds(reversed_boxes(more,box <> outer),reversed_digits(digits,later)) == prefix_holds(Con{box,more},digits) && BoxParts.box_holds(outer,later) : Bool},inside)
    case Con{+box,+more} Con{+extent,+done} Nil{}:
      Empty.absurd({BoxParts.box_holds(reversed_boxes(more,box <> outer),later) == prefix_holds(Con{box,more},Nil{}) && BoxParts.box_holds(outer,later) : Bool},below)

# A finished prefix steps the same source cells outer axis first as innermost
# first.
def offset_reversed(boxes: List<&2,BoxParts.BoxAxis>,done: List<&2,U32>,digits: List<&2,Nat>,+outer: List<&2,BoxParts.BoxAxis>,+later: List<&2,Nat>,+rest: List<&2,U32>,
  +inside: PrefixIn(boxes,done,rest),+below: Coordinates.DigitsBelow(done,digits)) ->
  {Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,outer),reversed_digits(digits,later)),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,outer)))) ==
    (prefix_offset(boxes,digits) + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32) : U32}:
  match boxes done digits:
    case Nil{} Nil{} Nil{}:
      Equal.sym(U32,(0 + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32),
        Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))),
        Arithmetic.add_zero_left(Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer)))))
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more} Con{+extent,+done} Con{+digit,+digits}:
      Held.Both{+same,Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+inner}}}}} = inside
      KernelGemm.Both{+lower,+below_more} = below
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      +step = (U32.from_nat(Nat.sub(digit,start)) * source_stride : U32)
      +others = prefix_offset(more,digits)
      +tail = Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer)))
      Equal.trans(U32,Digits.stepped_digits(BoxParts.box_within(reversed_boxes(more,box <> outer),reversed_digits(digits,digit <> later)),
          ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(more,box <> outer)))),
        (others + (step + tail : U32) : U32),((step + others : U32) + tail : U32),
        offset_reversed(more,done,digits,box <> outer,digit <> later,extent <> rest,inner,below_more),
        Equal.trans(U32,(others + (step + tail : U32) : U32),((others + step : U32) + tail : U32),((step + others : U32) + tail : U32),
          Equal.sym(U32,((others + step : U32) + tail : U32),(others + (step + tail : U32) : U32),Arithmetic.add_associative(others,step,tail)),
          Equal.cong(U32,U32,sum => (sum + tail : U32),(others + step : U32),(step + others : U32),Arithmetic.add_commutative(others,step))))
    case Nil{} Nil{} Con{+digit,+digits}:
      Empty.absurd({Digits.stepped_digits(BoxParts.box_within(outer,reversed_digits(digits,digit <> later)),ViewPart.move_steps(BoxParts.box_axes(outer))) ==
        (prefix_offset(Nil{},Con{digit,digits}) + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32) : U32},below)
    case Nil{} Con{+extent,+done} +digits:
      Empty.absurd({Digits.stepped_digits(BoxParts.box_within(outer,reversed_digits(digits,later)),ViewPart.move_steps(BoxParts.box_axes(outer))) ==
        (prefix_offset(Nil{},digits) + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32) : U32},inside)
    case Con{+box,+more} Nil{} +digits:
      Empty.absurd({Digits.stepped_digits(BoxParts.box_within(reversed_boxes(more,box <> outer),reversed_digits(digits,later)),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(more,box <> outer)))) ==
        (prefix_offset(Con{box,more},digits) + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32) : U32},inside)
    case Con{+box,+more} Con{+extent,+done} Nil{}:
      Empty.absurd({Digits.stepped_digits(BoxParts.box_within(reversed_boxes(more,box <> outer),later),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(more,box <> outer)))) ==
        (prefix_offset(Con{box,more},Nil{}) + Digits.stepped_digits(BoxParts.box_within(outer,later),ViewPart.move_steps(BoxParts.box_axes(outer))) : U32) : U32},below)

# The value a prefix gives a cell: the source cell at its first source cell
# plus offset when it holds the cell and the later axes reach the source at
# offset, else old.
def chosen(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,rest: Maybe<&2,U32>,found: Bool,+base: U32,+old: Element) -> Element:
  match rest found:
    case Some{+offset} True{}: value(context,(base + offset : U32))
    case rest found: old

def prefixes_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Prefix>,+digits: List<&2,Nat>,+rest: Maybe<&2,U32>,+old: Element) -> Element:
  match list:
    case Nil{}: old
    case Con{Prefix{+boxes,+source,+target},more}: prefixes_value(~Element,~Context,~value,context,more,digits,rest,chosen(~Element,~Context,~value,context,rest,prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old))

def final_step(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,found: Bool,+boxes: List<&2,BoxParts.BoxAxis>,+done: List<&2,U32>,+digits: List<&2,Nat>,+source: U32,+target: U32,+query: U32,+old: Element,
  +inside: BoxParts.BoxIn(reversed_boxes(boxes,[])),+inner: PrefixIn(boxes,done,[]),+below: Coordinates.DigitsBelow(done,digits),+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))),U32.to_nat(cap)) == True{} : Bool},
  +exact: {Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query)) == reversed_digits(digits,[]) : List<&2,Nat>},+holds: {BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))) == found : Bool}) ->
  {Bool.pick(Element,found,value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old) == chosen(~Element,~Context,~value,context,Some{0},found,(source + prefix_offset(boxes,digits) : U32),old) : Element}:
  match found:
    case True{}:
      +index_in = BoxParts.hit_below(reversed_boxes(boxes,[]),query,holds)
      Equal.cong(U32,Element,cell => value(context,cell),Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),
        Equal.trans(U32,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target),(source + Box.stepped_sum(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)) : U32),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),ViewPart.cell_sum(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target,cap,cap_ok,BoxParts.box_room(reversed_boxes(boxes,[]),inside,cap,cap_ok,piece_room),index_in),
        Equal.trans(U32,(source + Box.stepped_sum(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)) : U32),(source + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),Equal.cong(U32,U32,sum => (source + sum : U32),Box.stepped_sum(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)),Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))),Digits.stepped_coordinates(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query))),
        Equal.trans(U32,(source + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),(source + Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),Equal.cong(List<&2,Nat>,U32,list => (source + Digits.stepped_digits(list,ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.hit_index(reversed_boxes(boxes,[]),query)),BoxParts.box_within(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),Digits.coordinates_value(ViewPart.move_extents(BoxParts.box_axes(reversed_boxes(boxes,[]))),BoxParts.box_within(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),BoxParts.holds_below(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query)),holds))),
        Equal.trans(U32,(source + Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),(source + Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),reversed_digits(digits,[])),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),Equal.cong(List<&2,Nat>,U32,list => (source + Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),list),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query)),reversed_digits(digits,[]),exact),
        Equal.trans(U32,(source + Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),reversed_digits(digits,[])),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))) : U32),(source + (prefix_offset(boxes,digits) + 0 : U32) : U32),((source + prefix_offset(boxes,digits) : U32) + 0 : U32),Equal.cong(U32,U32,sum => (source + sum : U32),Digits.stepped_digits(BoxParts.box_within(reversed_boxes(boxes,[]),reversed_digits(digits,[])),ViewPart.move_steps(BoxParts.box_axes(reversed_boxes(boxes,[])))),(prefix_offset(boxes,digits) + 0 : U32),offset_reversed(boxes,done,digits,[],[],[],inner,below)),
        Equal.sym(U32,((source + prefix_offset(boxes,digits) : U32) + 0 : U32),(source + (prefix_offset(boxes,digits) + 0 : U32) : U32),Arithmetic.add_associative(source,prefix_offset(boxes,digits),0))))))))
    case False{}: {==}

# Finished prefixes give a cell the value they give it innermost first.
def final_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,list: List<&2,Prefix>,+done: List<&2,U32>,+digits: List<&2,Nat>,+query: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(reversed_extents(done,[]))),U32.to_nat(cap)) == True{} : Bool},
  +exact: {reversed_digits(digits,[]) == Targets.cell_coordinates(reversed_extents(done,[]),U32.to_nat(query)) : List<&2,Nat>},+below: Coordinates.DigitsBelow(done,digits),+old: Element,+held: PrefixesIn(list,done,[])) ->
  {boxes_value(~Element,~Context,~value,context,list,query,old) == prefixes_value(~Element,~Context,~value,context,list,digits,Some{0},old) : Element}:
  match list:
    case Nil{}: {==}
    case Con{Prefix{+boxes,+source,+target},+rest}:
      Held.Both{+own,Held.Both{+placed,+held_rest}} = held
      +same = prefix_box_extents(boxes,done,[],own)
      +exact_box = Equal.trans(List<&2,Nat>,Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query)),Targets.cell_coordinates(reversed_extents(done,[]),U32.to_nat(query)),reversed_digits(digits,[]),
        Equal.cong(List<&2,U32>,List<&2,Nat>,extents => Targets.cell_coordinates(extents,U32.to_nat(query)),BoxParts.box_extents(reversed_boxes(boxes,[])),reversed_extents(done,[]),same),
        Equal.sym(List<&2,Nat>,reversed_digits(digits,[]),Targets.cell_coordinates(reversed_extents(done,[]),U32.to_nat(query)),exact))
      +holds_same = Equal.trans(Bool,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),BoxParts.box_holds(reversed_boxes(boxes,[]),reversed_digits(digits,[])),prefix_holds(boxes,digits),
        Equal.cong(List<&2,Nat>,Bool,list => BoxParts.box_holds(reversed_boxes(boxes,[]),list),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query)),reversed_digits(digits,[]),exact_box),
        Equal.trans(Bool,BoxParts.box_holds(reversed_boxes(boxes,[]),reversed_digits(digits,[])),prefix_holds(boxes,digits) && True{},prefix_holds(boxes,digits),
          holds_reversed(boxes,done,digits,[],[],[],own,below),and_true(prefix_holds(boxes,digits))))
      +step = final_step(~Element,~Context,~value,context,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),boxes,done,digits,source,target,query,old,prefix_box_in(boxes,done,[],own,Unit{}),own,below,cap,cap_ok,
        {%Equal.sym(List<&2,U32>,BoxParts.box_extents(reversed_boxes(boxes,[])),reversed_extents(done,[]),same) : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool} piece_room : {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(reversed_boxes(boxes,[])))),U32.to_nat(cap)) == True{} : Bool}},exact_box,{==})
      Equal.trans(Element,boxes_value(~Element,~Context,~value,context,rest,query,Bool.pick(Element,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old)),boxes_value(~Element,~Context,~value,context,rest,query,chosen(~Element,~Context,~value,context,Some{0},prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old)),prefixes_value(~Element,~Context,~value,context,rest,digits,Some{0},chosen(~Element,~Context,~value,context,Some{0},prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old)),
        Equal.cong(Element,Element,acc => boxes_value(~Element,~Context,~value,context,rest,query,acc),Bool.pick(Element,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old),chosen(~Element,~Context,~value,context,Some{0},prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old),
          {%holds_same : {Bool.pick(Element,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old) == chosen(~Element,~Context,~value,context,Some{0},_,(source + prefix_offset(boxes,digits) : U32),old) : Element} step :
            {Bool.pick(Element,BoxParts.box_holds(reversed_boxes(boxes,[]),Targets.cell_coordinates(BoxParts.box_extents(reversed_boxes(boxes,[])),U32.to_nat(query))),value(context,Part.cell_source(BoxParts.box_axes(reversed_boxes(boxes,[])),BoxParts.hit_index(reversed_boxes(boxes,[]),query),source,target)),old) == chosen(~Element,~Context,~value,context,Some{0},prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old) : Element}}),
        final_value(~Element,~Context,~value,context,rest,done,digits,query,cap,cap_ok,piece_room,exact,below,chosen(~Element,~Context,~value,context,Some{0},prefix_holds(boxes,digits),(source + prefix_offset(boxes,digits) : U32),old),held_rest))

# ---- One more axis ----

# The source offset of the last box axis of an axis that holds digit, if one
# does.
def last_value(boxes: List<&2,BoxParts.BoxAxis>,+digit: Nat,+found: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match boxes:
    case Nil{}: found
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,target_stride,target_offset},+start,extent},rest}:
      last_value(rest,digit,Bool.pick(Maybe<&2,U32>,Nat.is_le(start,digit) && Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))),Some{(source_offset + (U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) : U32)},found))

def attached(rest: Maybe<&2,U32>,found: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match rest:
    case None{}: None{}
    case Some{+offset}:
      match found:
        case None{}: None{}
        case Some{+axis}: Some{(axis + offset : U32)}

def combined(found: Maybe<&2,U32>,rest: Maybe<&2,U32>) -> Maybe<&2,U32>:
  attached(rest,found)

# The values that the extensions of one prefix give a cell, box axis after box
# axis.
def cuts_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+digit: Nat,+digits: List<&2,Nat>,+rest: Maybe<&2,U32>,+old: Element) -> Element:
  match boxes:
    case Nil{}: old
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},more}:
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      cuts_value(~Element,~Context,~value,context,more,inner,source,digit,digits,rest,chosen(~Element,~Context,~value,context,rest,prefix_holds(box <> inner,digit <> digits),((source + source_offset : U32) + prefix_offset(box <> inner,digit <> digits) : U32),old))

def prefix_extended_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+target: U32,+later: List<&2,Prefix>,+digit: Nat,+digits: List<&2,Nat>,
  +rest: Maybe<&2,U32>,+old: Element) ->
  {prefixes_value(~Element,~Context,~value,context,prefix_extended(boxes,inner,source,target,later),digit <> digits,rest,old) ==
    prefixes_value(~Element,~Context,~value,context,later,digit <> digits,rest,cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,old)) : Element}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      prefix_extended_value(~Element,~Context,~value,context,more,inner,source,target,later,digit,digits,rest,
        chosen(~Element,~Context,~value,context,rest,prefix_holds(box <> inner,digit <> digits),((source + source_offset : U32) + prefix_offset(box <> inner,digit <> digits) : U32),old))

def regroup(+s: U32,+a: U32,+c: U32,+p: U32,+o: U32) -> {(((s + a : U32) + (c + p : U32) : U32) + o : U32) == ((s + p : U32) + ((a + c : U32) + o : U32) : U32) : U32}:
  Equal.trans(U32,(((s + a : U32) + (c + p : U32) : U32) + o : U32),(((s + a : U32) + (p + c : U32) : U32) + o : U32),((s + p : U32) + ((a + c : U32) + o : U32) : U32),Equal.cong(U32,U32,sum => (((s + a : U32) + sum : U32) + o : U32),(c + p : U32),(p + c : U32),Arithmetic.add_commutative(c,p)),
        Equal.trans(U32,(((s + a : U32) + (p + c : U32) : U32) + o : U32),((((s + a : U32) + p : U32) + c : U32) + o : U32),((s + p : U32) + ((a + c : U32) + o : U32) : U32),Equal.cong(U32,U32,sum => (sum + o : U32),((s + a : U32) + (p + c : U32) : U32),(((s + a : U32) + p : U32) + c : U32),Equal.sym(U32,(((s + a : U32) + p : U32) + c : U32),((s + a : U32) + (p + c : U32) : U32),Arithmetic.add_associative((s + a : U32),p,c))),
        Equal.trans(U32,((((s + a : U32) + p : U32) + c : U32) + o : U32),((((s + p : U32) + a : U32) + c : U32) + o : U32),((s + p : U32) + ((a + c : U32) + o : U32) : U32),Equal.cong(U32,U32,sum => ((sum + c : U32) + o : U32),((s + a : U32) + p : U32),((s + p : U32) + a : U32),Arithmetic.add_swap(s,a,p)),
        Equal.trans(U32,((((s + p : U32) + a : U32) + c : U32) + o : U32),(((s + p : U32) + (a + c : U32) : U32) + o : U32),((s + p : U32) + ((a + c : U32) + o : U32) : U32),Equal.cong(U32,U32,sum => (sum + o : U32),(((s + p : U32) + a : U32) + c : U32),((s + p : U32) + (a + c : U32) : U32),Arithmetic.add_associative((s + p : U32),a,c)),
        Arithmetic.add_associative((s + p : U32),(a + c : U32),o)))))

def step_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,rest: Maybe<&2,U32>,first: Bool,second: Bool,held: Bool,found: Maybe<&2,U32>,+source: U32,+source_offset: U32,+start: Nat,+digit: Nat,+source_stride: U32,
  +inner: List<&2,BoxParts.BoxAxis>,+digits: List<&2,Nat>,+old: Element) -> {chosen(~Element,~Context,~value,context,rest,first && (second && held),((source + source_offset : U32) + ((U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) + prefix_offset(inner,digits) : U32) : U32),chosen(~Element,~Context,~value,context,combined(found,rest),held,(source + prefix_offset(inner,digits) : U32),old)) == chosen(~Element,~Context,~value,context,combined(Bool.pick(Maybe<&2,U32>,first && second,Some{(source_offset + (U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) : U32)},found),rest),held,(source + prefix_offset(inner,digits) : U32),old) : Element}:
  match rest:
    case None{}: {==}
    case Some{+offset}:
      match first second held found:
        case True{} True{} True{} found:
          Equal.cong(U32,Element,cell => value(context,cell),(((source + source_offset : U32) + ((U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) + prefix_offset(inner,digits) : U32) : U32) + offset : U32),
            ((source + prefix_offset(inner,digits) : U32) + ((source_offset + (U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) : U32) + offset : U32) : U32),regroup(source,source_offset,(U32.from_nat(Nat.sub(digit,start)) * source_stride : U32),prefix_offset(inner,digits),offset))
        case True{} True{} False{} None{}: {==}
        case True{} True{} False{} Some{+axis}: {==}
        case True{} False{} held found: {==}
        case False{} second held found: {==}

def unchosen(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,rest: Maybe<&2,U32>,+held: Bool,+base: U32,+old: Element) -> {old == chosen(~Element,~Context,~value,context,combined(None{},rest),held,base,old) : Element}:
  match rest:
    case None{}: {==}
    case Some{+offset}: {==}

# The extensions of one prefix give a cell what the prefix gives it with the
# last box axis that holds the digit.
def cuts_chosen(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+source: U32,+digit: Nat,+digits: List<&2,Nat>,+rest: Maybe<&2,U32>,+found: Maybe<&2,U32>,+old: Element) ->
  {cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,chosen(~Element,~Context,~value,context,combined(found,rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old)) ==
    chosen(~Element,~Context,~value,context,combined(last_value(boxes,digit,found),rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old) : Element}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      +first = Nat.is_le(start,digit)
      +second = Nat.is_lt(digit,Nat.add(start,U32.to_nat(count)))
      +next = Bool.pick(Maybe<&2,U32>,first && second,Some{(source_offset + (U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) : U32)},found)
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Equal.trans(Element,cuts_value(~Element,~Context,~value,context,more,inner,source,digit,digits,rest,
          chosen(~Element,~Context,~value,context,rest,first && (second && prefix_holds(inner,digits)),((source + source_offset : U32) + ((U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) + prefix_offset(inner,digits) : U32) : U32),chosen(~Element,~Context,~value,context,combined(found,rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old))),
        cuts_value(~Element,~Context,~value,context,more,inner,source,digit,digits,rest,chosen(~Element,~Context,~value,context,combined(next,rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old)),
        chosen(~Element,~Context,~value,context,combined(last_value(more,digit,next),rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old),
        Equal.cong(Element,Element,acc => cuts_value(~Element,~Context,~value,context,more,inner,source,digit,digits,rest,acc),
          chosen(~Element,~Context,~value,context,rest,first && (second && prefix_holds(inner,digits)),((source + source_offset : U32) + ((U32.from_nat(Nat.sub(digit,start)) * source_stride : U32) + prefix_offset(inner,digits) : U32) : U32),chosen(~Element,~Context,~value,context,combined(found,rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old)),
          chosen(~Element,~Context,~value,context,combined(next,rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old),
          step_value(~Element,~Context,~value,context,rest,first,second,prefix_holds(inner,digits),found,source,source_offset,start,digit,source_stride,inner,digits,old)),
        cuts_chosen(~Element,~Context,~value,context,more,inner,source,digit,digits,rest,next,old))

# Extending every prefix by the box axes of one more axis gives a cell what
# the prefixes give it with the last box axis that holds its digit.
def extended_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,prefixes: List<&2,Prefix>,+boxes: List<&2,BoxParts.BoxAxis>,+digit: Nat,+digits: List<&2,Nat>,+rest: Maybe<&2,U32>,+old: Element) ->
  {prefixes_value(~Element,~Context,~value,context,prefixes_extended(prefixes,boxes),digit <> digits,rest,old) ==
    prefixes_value(~Element,~Context,~value,context,prefixes,digits,combined(last_value(boxes,digit,None{}),rest),old) : Element}:
  match prefixes:
    case Nil{}: {==}
    case Con{Prefix{+inner,+source,+target},+more}:
      +later = prefixes_extended(more,boxes)
      +whole = combined(last_value(boxes,digit,None{}),rest)
      +picked = chosen(~Element,~Context,~value,context,whole,prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old)
      Equal.trans(Element,prefixes_value(~Element,~Context,~value,context,prefix_extended(boxes,inner,source,target,later),digit <> digits,rest,old),
        prefixes_value(~Element,~Context,~value,context,later,digit <> digits,rest,cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,old)),
        prefixes_value(~Element,~Context,~value,context,more,digits,whole,picked),
        prefix_extended_value(~Element,~Context,~value,context,boxes,inner,source,target,later,digit,digits,rest,old),
        Equal.trans(Element,prefixes_value(~Element,~Context,~value,context,later,digit <> digits,rest,cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,old)),
          prefixes_value(~Element,~Context,~value,context,later,digit <> digits,rest,picked),prefixes_value(~Element,~Context,~value,context,more,digits,whole,picked),
          Equal.cong(Element,Element,acc => prefixes_value(~Element,~Context,~value,context,later,digit <> digits,rest,acc),cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,old),picked,
            {%Equal.sym(Element,old,chosen(~Element,~Context,~value,context,combined(None{},rest),prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old),unchosen(~Element,~Context,~value,context,rest,prefix_holds(inner,digits),(source + prefix_offset(inner,digits) : U32),old)) :
                {cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,_) == picked : Element}
              cuts_chosen(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,None{},old) :
              {cuts_value(~Element,~Context,~value,context,boxes,inner,source,digit,digits,rest,old) == picked : Element}}),
          extended_value(~Element,~Context,~value,context,more,boxes,digit,digits,rest,picked)))

# ---- Every axis ----

# The source offset that the cell of piece digits reads, if on every axis a
# box axis holds its digit.
def placed_offset(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  digits: List<&2,Nat>) -> Maybe<&2,U32>:
  match segments coordinates extents source_strides piece_strides digits:
    case Con{own,segments} Con{+low,coordinates} Con{+extent,extents} Con{+source_stride,source_strides} Con{+piece_stride,piece_strides} Con{+digit,digits}:
      combined(last_value(axis_boxes(own,low,extent,source_stride,piece_stride),digit,None{}),placed_offset(segments,coordinates,extents,source_strides,piece_strides,digits))
    case segments coordinates extents source_strides piece_strides digits: Some{0}

def ended_digits(later: List<&2,Nat>,+digits: List<&2,Nat>,+ended: Coordinates.DigitsBelow(Nil{},later)) -> {reversed_digits(later,digits) == digits : List<&2,Nat>}:
  match later:
    case Nil{}: {==}
    case Con{+digit,+more}: Empty.absurd({reversed_digits(more,digit <> digits) == digits : List<&2,Nat>},ended)

def ended_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+prefixes: List<&2,Prefix>,+digits: List<&2,Nat>,+later: List<&2,Nat>,+extents: List<&2,U32>,+old: Element,+ended: {extents == [] : List<&2,U32>},
  +below: Coordinates.DigitsBelow(extents,later)) ->
  {prefixes_value(~Element,~Context,~value,context,prefixes,reversed_digits(later,digits),Some{0},old) == prefixes_value(~Element,~Context,~value,context,prefixes,digits,Some{0},old) : Element}:
  Equal.cong(List<&2,Nat>,Element,list => prefixes_value(~Element,~Context,~value,context,prefixes,list,Some{0},old),reversed_digits(later,digits),digits,
    ended_digits(later,digits,{%ended : Coordinates.DigitsBelow(_,later) below : Coordinates.DigitsBelow(Nil{},later)}))

# Axis after axis, the parts of a placement give a cell what the prefixes give
# it with the source offset of the later axes.
def placed_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  +prefixes: List<&2,Prefix>,+digits: List<&2,Nat>,+later: List<&2,Nat>,+old: Element,+axes: AxesBoxed(segments,coordinates,extents,source_strides,piece_strides),
  +below: Coordinates.DigitsBelow(extents,later)) ->
  {prefixes_value(~Element,~Context,~value,context,placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,prefixes),reversed_digits(later,digits),Some{0},old) ==
    prefixes_value(~Element,~Context,~value,context,prefixes,digits,placed_offset(segments,coordinates,extents,source_strides,piece_strides,later),old) : Element}:
  match segments:
    case Nil{}: ended_value(~Element,~Context,~value,context,prefixes,digits,later,extents,old,axes,below)
    case Con{+own,+segments}:
      match coordinates:
        case Nil{}: ended_value(~Element,~Context,~value,context,prefixes,digits,later,extents,old,axes,below)
        case Con{+low,+coordinates}:
          match extents:
            case Nil{}: ended_value(~Element,~Context,~value,context,prefixes,digits,later,[],old,axes,below)
            case Con{+extent,+extents}:
              match source_strides:
                case Nil{}: ended_value(~Element,~Context,~value,context,prefixes,digits,later,extent <> extents,old,axes,below)
                case Con{+source_stride,+source_strides}:
                  match piece_strides:
                    case Nil{}: ended_value(~Element,~Context,~value,context,prefixes,digits,later,extent <> extents,old,axes,below)
                    case Con{+piece_stride,+piece_strides}:
                      match later:
                        case Nil{}:
                          Empty.absurd({prefixes_value(~Element,~Context,~value,context,placed_prefixes(Con{own,segments},Con{low,coordinates},Con{extent,extents},Con{source_stride,source_strides},Con{piece_stride,piece_strides},prefixes),
                            digits,Some{0},old) == prefixes_value(~Element,~Context,~value,context,prefixes,digits,Some{0},old) : Element},below)
                        case Con{+digit,+more}:
                          Held.Both{+axis,+axes_more} = axes
                          KernelGemm.Both{+lower,+below_more} = below
                          +boxes = axis_boxes(own,low,extent,source_stride,piece_stride)
                          +offset = placed_offset(segments,coordinates,extents,source_strides,piece_strides,more)
                          Equal.trans(Element,prefixes_value(~Element,~Context,~value,context,placed_prefixes(segments,coordinates,extents,source_strides,piece_strides,prefixes_extended(prefixes,boxes)),
                              reversed_digits(more,digit <> digits),Some{0},old),
                            prefixes_value(~Element,~Context,~value,context,prefixes_extended(prefixes,boxes),digit <> digits,offset,old),
                            prefixes_value(~Element,~Context,~value,context,prefixes,digits,combined(last_value(boxes,digit,None{}),offset),old),
                            placed_value(~Element,~Context,~value,context,segments,coordinates,extents,source_strides,piece_strides,prefixes_extended(prefixes,boxes),digit <> digits,more,old,axes_more,below_more),
                            extended_value(~Element,~Context,~value,context,prefixes,boxes,digit,digits,offset,old))

# The value of a target cell: the source cell at offset, or old when an axis
# does not reach it.
def placed_result(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,offset: Maybe<&2,U32>,+old: Element) -> Element:
  match offset:
    case Some{+cell}: value(context,cell)
    case None{}: old

def first_result(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,offset: Maybe<&2,U32>,+old: Element) -> {chosen(~Element,~Context,~value,context,offset,True{},(0 + 0 : U32),old) == placed_result(~Element,~Context,~value,context,offset,old) : Element}:
  match offset:
    case Some{+cell}:
      Equal.cong(U32,Element,cell => value(context,cell),((0 + 0 : U32) + cell : U32),cell,
        Equal.trans(U32,((0 + 0 : U32) + cell : U32),(0 + cell : U32),cell,
          Equal.cong(U32,U32,first => (first + cell : U32),(0 + 0 : U32),0,Arithmetic.add_zero(0)),Arithmetic.add_zero_left(cell)))
    case None{}: {==}

# ---- From box axes to segments ----

# The source index the last segment of an axis covering position reads, if
# one does.
def segment_index(segments: List<&2,Tasks.Segment>,+position: U32,+found: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match segments:
    case Nil{}: found
    case Con{Tasks.Segment{+first,+count,+source,+step},rest}:
      segment_index(rest,position,Bool.pick(Maybe<&2,U32>,U32.is_le(first,position) && U32.is_lt(position,(first + count : U32)),
        Some{(source + (position - first) * step : U32)},found))

def strided(found: Maybe<&2,U32>,+stride: U32) -> Maybe<&2,U32>:
  match found:
    case None{}: None{}
    case Some{+index}: Some{(index * stride : U32)}

# The target coordinates of piece digits.
def positions(coordinates: List<&2,U32>,digits: List<&2,Nat>) -> List<&2,U32>:
  match coordinates digits:
    case Con{+low,rest} Con{+digit,more}: (low + U32.from_nat(digit) : U32) <> positions(rest,more)
    case coordinates digits: []

# The source cell that the target cell at coordinates reads: on every axis the
# source index of the last segment covering the coordinate times the source
# stride, if every axis has one.
def segments_cell(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,source_strides: List<&2,U32>) -> Maybe<&2,U32>:
  match segments coordinates source_strides:
    case Con{own,segments} Con{+position,coordinates} Con{+stride,source_strides}:
      combined(strided(segment_index(own,position,None{}),stride),segments_cell(segments,coordinates,source_strides))
    case segments coordinates source_strides: Some{0}

# Every segment ends inside cap.
def SegmentsIn(segments: List<&2,Tasks.Segment>,+cap: U32) -> Data:
  match segments:
    case Nil{}: Unit
    case Con{Tasks.Segment{+first,+count,source,step},rest}: Held.Both<{Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},SegmentsIn(rest,cap)>

# Every axis of the piece ends inside cap and has segments that do, and the
# lists end together with the extents.
def AxesIn(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,+cap: U32) -> Data:
  match segments coordinates extents source_strides:
    case Con{own,segments} Con{+low,coordinates} Con{+extent,extents} Con{+source_stride,source_strides}:
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},Held.Both<SegmentsIn(own,cap),AxesIn(segments,coordinates,extents,source_strides,cap)>>
    case segments coordinates extents source_strides: {extents == [] : List<&2,U32>}

def top_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((first + count : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}:
  Arithmetic.add(first,count,cap,cap_ok,segment_room)

def end_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.min((first + count : U32),(low + extent : U32))) == Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))) : Nat}:
  %top_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room) : {U32.to_nat(U32.min((first + count : U32),(low + extent : U32))) == Nat.min(_,Nat.add(U32.to_nat(low),U32.to_nat(extent))) : Nat}
  %Arithmetic.add(low,extent,cap,cap_ok,piece_room) : {U32.to_nat(U32.min((first + count : U32),(low + extent : U32))) == Nat.min(U32.to_nat((first + count : U32)),_) : Nat}
  Extrema.minimum((first + count : U32),(low + extent : U32))

def max_room(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(cap)) == True{} : Bool}:
  NatExtrema.maximum_upper(U32.to_nat(first),U32.to_nat(low),U32.to_nat(cap),Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.le_add(U32.to_nat(first),U32.to_nat(count)),segment_room),Order.transitive(U32.to_nat(low),Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap),Order.le_add(U32.to_nat(low),U32.to_nat(extent)),piece_room))

def offset_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((U32.max(first,low) - low : U32)) == Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)) : Nat}:
  %Extrema.maximum(first,low) : {U32.to_nat((U32.max(first,low) - low : U32)) == Nat.sub(_,U32.to_nat(low)) : Nat}
  Arithmetic.subtract(U32.max(first,low),low,cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(U32.max(first,low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),Extrema.maximum(first,low)) : {Nat.is_le(U32.to_nat(low),_) == True{} : Bool} NatExtrema.maximum_right(U32.to_nat(first),U32.to_nat(low)) : {Nat.is_le(U32.to_nat(low),U32.to_nat(U32.max(first,low))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.max(first,low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),Extrema.maximum(first,low)) : {Nat.is_le(Nat.sub(_,U32.to_nat(low)),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(cap),Order.sub_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),max_room(first,count,low,extent,cap,cap_ok,segment_room,piece_room)) : {Nat.is_le(Nat.sub(U32.to_nat(U32.max(first,low)),U32.to_nat(low)),U32.to_nat(cap)) == True{} : Bool}})

def count_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+apart: {Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) == True{} : Bool}) -> {U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)) == Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))) : Nat}:
  %Extrema.maximum(first,low) : {U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)) == Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),_) : Nat}
  %end_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room) : {U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)) == Nat.sub(_,U32.to_nat(U32.max(first,low))) : Nat}
  Arithmetic.subtract(U32.min((first + count : U32),(low + extent : U32)),U32.max(first,low),cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(U32.max(first,low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),Extrema.maximum(first,low)) : {Nat.is_le(_,U32.to_nat(U32.min((first + count : U32),(low + extent : U32)))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.min((first + count : U32),(low + extent : U32))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),end_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)) : {Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),_) == True{} : Bool}
      Bounds.strict_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),apart) : {Nat.is_le(U32.to_nat(U32.max(first,low)),U32.to_nat(U32.min((first + count : U32),(low + extent : U32)))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.max(first,low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),Extrema.maximum(first,low)) : {Nat.is_le(Nat.sub(U32.to_nat(U32.min((first + count : U32),(low + extent : U32))),_),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.min((first + count : U32),(low + extent : U32))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),end_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)) : {Nat.is_le(Nat.sub(_,Nat.max(U32.to_nat(first),U32.to_nat(low))),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap),Order.sub_le(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),
        Order.transitive(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),NatExtrema.minimum_left(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),segment_room)) :
      {Nat.is_le(Nat.sub(U32.to_nat(U32.min((first + count : U32),(low + extent : U32))),U32.to_nat(U32.max(first,low))),U32.to_nat(cap)) == True{} : Bool}})

def apart_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) : Bool}:
  %Extrema.maximum(first,low) : {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == Nat.is_lt(_,Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) : Bool}
  %end_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room) : {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == Nat.is_lt(U32.to_nat(U32.max(first,low)),_) : Bool}
  Comparison.lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32)))

def position_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) -> {U32.to_nat((low + U32.from_nat(digit) : U32)) == Nat.add(U32.to_nat(low),digit) : Nat}:
  +back = Bounds.roundtrip(digit,cap,cap_ok,Order.transitive(digit,U32.to_nat(extent),U32.to_nat(cap),Bounds.strict_le(digit,U32.to_nat(extent),inside),Order.transitive(U32.to_nat(extent),Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap),Order.right_le(U32.to_nat(low),U32.to_nat(extent)),piece_room)))
  %back : {U32.to_nat((low + U32.from_nat(digit) : U32)) == Nat.add(U32.to_nat(low),_) : Nat}
  Arithmetic.add(low,U32.from_nat(digit),cap,cap_ok,
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(digit)),digit,back) : {Nat.is_le(Nat.add(U32.to_nat(low),_),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap),
        {%Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.is_le(digit,U32.to_nat(extent)),Order.le_offset(U32.to_nat(low),digit,U32.to_nat(extent))) : {_ == True{} : Bool}
          Bounds.strict_le(digit,U32.to_nat(extent),inside) : {Nat.is_le(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool}},piece_room) : {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(U32.from_nat(digit))),U32.to_nat(cap)) == True{} : Bool}})

def cover_word(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) -> {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))) : Bool}:
  %top_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room) : {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),_) : Bool}
  %position_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside) : {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == Nat.is_le(U32.to_nat(first),_) && Nat.is_lt(_,U32.to_nat((first + count : U32))) : Bool}
  %Comparison.le(first,(low + U32.from_nat(digit) : U32)) : {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == _ && Nat.is_lt(U32.to_nat((low + U32.from_nat(digit) : U32)),U32.to_nat((first + count : U32))) : Bool}
  %Comparison.lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && _ : Bool}
  {==}

def and_false(value: Bool) -> {value && False{} == False{} : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

def le_false_upward(found: Bool,+low: Nat,+high: Nat,+value: Nat,+ordered: {Nat.is_le(low,high) == True{} : Bool},+missed: {Nat.is_le(low,value) == False{} : Bool},
  +by_value: {Nat.is_le(high,value) == found : Bool}) -> {found == False{} : Bool}:
  match found:
    case False{}: {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(low,value),True{},Equal.sym(Bool,Nat.is_le(low,value),False{},missed),
        Order.transitive(low,high,value,ordered,by_value))))

def lt_false_downward(found: Bool,+low: Nat,+high: Nat,+value: Nat,+ordered: {Nat.is_le(low,high) == True{} : Bool},+missed: {Nat.is_lt(value,high) == False{} : Bool},
  +by_value: {Nat.is_lt(value,low) == found : Bool}) -> {found == False{} : Bool}:
  match found:
    case False{}: {==}
    case True{}:
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(value,high),True{},Equal.sym(Bool,Nat.is_lt(value,high),False{},missed),
        Bounds.nat_trans(value,low,high,by_value,ordered))))

# Inside the piece a coordinate lies between the larger start and the smaller
# end exactly when the segment covers it.
def window_cover(found_low: Bool,found_high: Bool,+first: Nat,+top: Nat,+low: Nat,+low_top: Nat,+value: Nat,+low_ok: {Nat.is_le(low,value) == True{} : Bool},+high_ok: {Nat.is_lt(value,low_top) == True{} : Bool},
  +by_low: {Nat.is_le(first,value) == found_low : Bool},+by_high: {Nat.is_lt(value,top) == found_high : Bool}) ->
  {Nat.is_le(Nat.max(first,low),value) && Nat.is_lt(value,Nat.min(top,low_top)) == found_low && found_high : Bool}:
  match found_low found_high:
    case True{} True{}:
      %Equal.sym(Bool,Nat.is_le(Nat.max(first,low),value),True{},NatExtrema.maximum_upper(first,low,value,by_low,low_ok)) :
        {_ && Nat.is_lt(value,Nat.min(top,low_top)) == True{} : Bool}
      NatExtrema.minimum_strict_lower(top,low_top,value,by_high,high_ok)
    case False{} found_high:
      %Equal.sym(Bool,Nat.is_le(Nat.max(first,low),value),False{},
        le_false_upward(Nat.is_le(Nat.max(first,low),value),first,Nat.max(first,low),value,NatExtrema.maximum_left(first,low),by_low,{==})) :
        {_ && Nat.is_lt(value,Nat.min(top,low_top)) == False{} : Bool}
      {==}
    case True{} False{}:
      %Equal.sym(Bool,Nat.is_lt(value,Nat.min(top,low_top)),False{},
        lt_false_downward(Nat.is_lt(value,Nat.min(top,low_top)),Nat.min(top,low_top),top,value,NatExtrema.minimum_left(top,low_top),by_high,{==})) :
        {Nat.is_le(Nat.max(first,low),value) && _ == False{} : Bool}
      and_false(Nat.is_le(Nat.max(first,low),value))

# A box axis of a segment holds a digit exactly when the segment covers its
# position.
def box_cover(+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+apart: {Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) == True{} : Bool}) -> {Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)))) == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : Bool}:
  +low_max = NatExtrema.maximum_right(U32.to_nat(first),U32.to_nat(low))
  +max_min = Bounds.strict_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),apart)
  +low_min = Order.transitive(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),low_max,max_min)
  +restored_max = Order.restore_shift(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),low_max)
  +restored_min = Order.restore_shift(U32.to_nat(low),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),low_min)
  +span = Order.cancel_sum(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),
    Equal.trans(Nat,Nat.add(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Addition.associative(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),restored_max),
        Equal.trans(Nat,Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Order.restore_shift(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),max_min),
        Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),restored_min)))))
  +lower = Equal.trans(Bool,Nat.is_le(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit),Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.add(U32.to_nat(low),digit)),Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),digit)),
    Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.add(U32.to_nat(low),digit)),Nat.is_le(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit),Order.le_offset(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit)),
    Equal.cong(Nat,Bool,start => Nat.is_le(start,Nat.add(U32.to_nat(low),digit)),Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),restored_max))
  +upper = Equal.trans(Bool,Nat.is_lt(digit,Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Nat.is_lt(digit,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),
    Equal.cong(Nat,Bool,end => Nat.is_lt(digit,end),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),span),
    Equal.trans(Bool,Nat.is_lt(digit,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)))),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),
      Equal.sym(Bool,Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)))),Nat.is_lt(digit,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Order.lt_offset(U32.to_nat(low),digit,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)))),
      Equal.cong(Nat,Bool,end => Nat.is_lt(Nat.add(U32.to_nat(low),digit),end),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),restored_min)))
  +window = window_cover(Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(low),Nat.add(U32.to_nat(low),U32.to_nat(extent)),Nat.add(U32.to_nat(low),digit),Order.le_add(U32.to_nat(low),digit),
    {%Equal.sym(Bool,Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.is_lt(digit,U32.to_nat(extent)),Order.lt_offset(U32.to_nat(low),digit,U32.to_nat(extent))) : {_ == True{} : Bool} inside : {Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool}},{==},{==})
  %Equal.sym(Nat,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),count_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,apart)) : {Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),_)) == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : Bool}
  %Equal.sym(Nat,U32.to_nat((U32.max(first,low) - low : U32)),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),offset_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)) : {Nat.is_le(_,digit) && Nat.is_lt(digit,Nat.add(_,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))) == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : Bool}
  %Equal.sym(Bool,Nat.is_le(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit),Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),digit)),lower) : {_ && Nat.is_lt(digit,Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))) == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : Bool}
  %Equal.sym(Bool,Nat.is_lt(digit,Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),upper) : {Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),digit)) && _ == U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) : Bool}
  Equal.trans(Bool,Nat.is_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),window,Equal.sym(Bool,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),cover_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside)))

# A segment without a box axis in the piece does not cover a position in it.
def uncovered(found: Bool,+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+missed: {Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) == False{} : Bool},+by_cover: {Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))) == found : Bool}) -> {found == False{} : Bool}:
  match found:
    case False{}: {==}
    case True{}:
      +covered_low = BoxParts.and_first(Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),by_cover)
      +covered_high = BoxParts.and_second(Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),by_cover)
      +high_ok = {%Equal.sym(Bool,Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.is_lt(digit,U32.to_nat(extent)),Order.lt_offset(U32.to_nat(low),digit,U32.to_nat(extent))) : {_ == True{} : Bool} inside : {Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool}}
      +before = Order.le_lt_trans(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),digit),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),NatExtrema.maximum_upper(U32.to_nat(first),U32.to_nat(low),Nat.add(U32.to_nat(low),digit),covered_low,Order.le_add(U32.to_nat(low),digit)),
        NatExtrema.minimum_strict_lower(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)),Nat.add(U32.to_nat(low),digit),covered_high,high_ok))
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),True{},
        Equal.sym(Bool,Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),False{},missed),before)))

# A box axis of a segment that holds a digit reads the source cell of the
# segment's index at the position, times the source stride.
def box_contribution(+source: U32,+step: U32,+source_stride: U32,+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+apart: {Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) == True{} : Bool},+covered: {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == True{} : Bool}) -> {(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32) == ((source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32) * source_stride : U32) : U32}:
  +first_max = NatExtrema.maximum_left(U32.to_nat(first),U32.to_nat(low))
  +low_max = NatExtrema.maximum_right(U32.to_nat(first),U32.to_nat(low))
  +covered_nat = Equal.trans(Bool,Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),True{},Equal.sym(Bool,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),cover_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside)),covered)
  +covered_low = BoxParts.and_first(Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)),Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),covered_nat)
  +max_position = NatExtrema.maximum_upper(U32.to_nat(first),U32.to_nat(low),Nat.add(U32.to_nat(low),digit),covered_low,Order.le_add(U32.to_nat(low),digit))
  +reach = Equal.trans(Bool,Nat.is_le(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit),Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.add(U32.to_nat(low),digit)),True{},
    Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.add(U32.to_nat(low),digit)),Nat.is_le(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit),Order.le_offset(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit)),
    {%Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),Order.restore_shift(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),low_max)) : {Nat.is_le(_,Nat.add(U32.to_nat(low),digit)) == True{} : Bool} max_position : {Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.add(U32.to_nat(low),digit)) == True{} : Bool}})
  +total = Equal.trans(Nat,Nat.add(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),Nat.add(Nat.add(U32.to_nat(first),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(low),digit),Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),Addition.associative(U32.to_nat(first),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(first),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(low),digit),Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(first),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),Nat.max(U32.to_nat(first),U32.to_nat(low)),Order.restore_shift(U32.to_nat(first),Nat.max(U32.to_nat(first),U32.to_nat(low)),first_max)),
        Equal.trans(Nat,Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(low),digit),Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),Order.restore_shift(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),low_max))),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),Nat.add(U32.to_nat(low),digit),Addition.associative(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),
        Equal.cong(Nat,Nat,sum => Nat.add(U32.to_nat(low),sum),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),digit,Order.restore_shift(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),digit,reach))))))
  +difference = Equal.trans(Nat,Nat.sub(Nat.add(U32.to_nat(low),digit),U32.to_nat(first)),Nat.sub(Nat.add(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),U32.to_nat(first)),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),
    Equal.cong(Nat,Nat,sum => Nat.sub(sum,U32.to_nat(first)),Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),Equal.sym(Nat,Nat.add(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),Nat.add(U32.to_nat(low),digit),total)),Order.subtract_shift(U32.to_nat(first),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))))
  +start_part = {%Extrema.maximum(first,low) : {(U32.max(first,low) - first : U32) == U32.from_nat(Nat.sub(_,U32.to_nat(first))) : U32}
    Arithmetic.native_subtract(U32.max(first,low),first,{%Equal.sym(Nat,U32.to_nat(U32.max(first,low)),Nat.max(U32.to_nat(first),U32.to_nat(low)),Extrema.maximum(first,low)) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} first_max : {Nat.is_le(U32.to_nat(first),U32.to_nat(U32.max(first,low))) == True{} : Bool}}) :
    {(U32.max(first,low) - first : U32) == U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))) : U32}}
  +position_part = {%position_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside) : {((low + U32.from_nat(digit) : U32) - first : U32) == U32.from_nat(Nat.sub(_,U32.to_nat(first))) : U32}
    Arithmetic.native_subtract((low + U32.from_nat(digit) : U32),first,{%Equal.sym(Nat,U32.to_nat((low + U32.from_nat(digit) : U32)),Nat.add(U32.to_nat(low),digit),position_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside)) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} covered_low : {Nat.is_le(U32.to_nat(first),U32.to_nat((low + U32.from_nat(digit) : U32))) == True{} : Bool}}) :
    {((low + U32.from_nat(digit) : U32) - first : U32) == U32.from_nat(Nat.sub(Nat.add(U32.to_nat(low),digit),U32.to_nat(first))) : U32}}
  +parts = Equal.trans(U32,((low + U32.from_nat(digit) : U32) - first : U32),U32.from_nat(Nat.sub(Nat.add(U32.to_nat(low),digit),U32.to_nat(first))),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32),position_part,
        Equal.trans(U32,U32.from_nat(Nat.sub(Nat.add(U32.to_nat(low),digit),U32.to_nat(first))),U32.from_nat(Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32),Equal.cong(Nat,U32,U32.from_nat,Nat.sub(Nat.add(U32.to_nat(low),digit),U32.to_nat(first)),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))),difference),
        Equal.trans(U32,U32.from_nat(Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),(U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))) + U32.from_nat(Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))) : U32),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32),Equal.sym(U32,(U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))) + U32.from_nat(Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))) : U32),U32.from_nat(Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),WordArithmetic.native_add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first)),Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))))),
        Equal.trans(U32,(U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))) + U32.from_nat(Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))) : U32),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))) : U32),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32),Equal.cong(U32,U32,word => (word + U32.from_nat(Nat.sub(digit,Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)))) : U32),U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),(U32.max(first,low) - first : U32),Equal.sym(U32,(U32.max(first,low) - first : U32),U32.from_nat(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(first))),start_part)),
        Equal.cong(Nat,U32,offset => ((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,offset)) : U32),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),U32.to_nat((U32.max(first,low) - low : U32)),Equal.sym(Nat,U32.to_nat((U32.max(first,low) - low : U32)),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),offset_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)))))))
  Equal.sym(U32,((source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),
    Equal.trans(U32,((source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32) * source_stride : U32),((source + ((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32) * step : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),Equal.cong(U32,U32,index => ((source + index * step : U32) * source_stride : U32),((low + U32.from_nat(digit) : U32) - first : U32),((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32),parts),
        Equal.trans(U32,((source + ((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32) * step : U32) * source_stride : U32),((source + ((U32.max(first,low) - first : U32) * step + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),Equal.cong(U32,U32,moved => ((source + moved : U32) * source_stride : U32),(((U32.max(first,low) - first : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) : U32) * step : U32),((U32.max(first,low) - first : U32) * step + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32),Arithmetic.multiply_additive((U32.max(first,low) - first : U32),U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))),step)),
        Equal.trans(U32,((source + ((U32.max(first,low) - first : U32) * step + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),Equal.cong(U32,U32,sum => (sum * source_stride : U32),(source + ((U32.max(first,low) - first : U32) * step + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) : U32),((source + (U32.max(first,low) - first : U32) * step : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32),Equal.sym(U32,((source + (U32.max(first,low) - first : U32) * step : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32),(source + ((U32.max(first,low) - first : U32) * step + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) : U32),Arithmetic.add_associative(source,((U32.max(first,low) - first : U32) * step : U32),(U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32)))),
        Equal.trans(U32,(((source + (U32.max(first,low) - first : U32) * step : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + (U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32) * source_stride : U32),(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),Arithmetic.multiply_additive((source + (U32.max(first,low) - first : U32) * step : U32),(U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step : U32),source_stride),
        Equal.cong(U32,U32,last => (((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + last : U32),(U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * step * source_stride : U32),(U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),Arithmetic.multiply_associative(U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))),step,source_stride)))))))

def picked_index(covers: Bool,+source: U32,+step: U32,+source_stride: U32,+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+apart: {Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))) == True{} : Bool},+found: Maybe<&2,U32>,+by_cover: {U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)) == covers : Bool}) ->
  {Bool.pick(Maybe<&2,U32>,covers,Some{(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32)},strided(found,source_stride)) == strided(Bool.pick(Maybe<&2,U32>,covers,Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride) : Maybe<&2,U32>}:
  match covers:
    case True{}: Equal.cong(U32,Maybe<&2,U32>,cell => Some{cell},(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32),((source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32) * source_stride : U32),box_contribution(source,step,source_stride,first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside,apart,by_cover))
    case False{}: {==}

# One segment: its box axis, if it has one in the piece, takes the source
# offset the segment gives the position.
def segment_step(picked: Bool,+source: U32,+step: U32,+source_stride: U32,+piece_stride: U32,+rest: List<&2,Tasks.Segment>,+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+found: Maybe<&2,U32>,
  +by_apart: {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == picked : Bool},
  +later: {last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride)) == strided(segment_index(rest,(low + U32.from_nat(digit) : U32),Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found)),source_stride) : Maybe<&2,U32>}) ->
  {last_value(Bool.pick(List<&2,BoxParts.BoxAxis>,picked,BoxParts.BoxAxis{Tasks.Cut{(U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32),(step * source_stride : U32),((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32),piece_stride,((U32.max(first,low) - low : U32) * piece_stride : U32)},U32.to_nat((U32.max(first,low) - low : U32)),extent} <> axis_boxes(rest,low,extent,source_stride,piece_stride),axis_boxes(rest,low,extent,source_stride,piece_stride)),digit,strided(found,source_stride)) == strided(segment_index(rest,(low + U32.from_nat(digit) : U32),Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found)),source_stride) : Maybe<&2,U32>}:
  match picked:
    case True{}:
      +apart = Equal.trans(Bool,Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),True{},Equal.sym(Bool,U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),apart_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)),by_apart)
      +same = box_cover(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside,apart)
      Equal.trans(Maybe<&2,U32>,last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,Bool.pick(Maybe<&2,U32>,Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)))),Some{(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32)},strided(found,source_stride))),
        last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride)),strided(segment_index(rest,(low + U32.from_nat(digit) : U32),Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found)),source_stride),
        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,acc => last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,acc),Bool.pick(Maybe<&2,U32>,Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)))),Some{(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32)},strided(found,source_stride)),
          strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride),
          {%Equal.sym(Bool,Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)))),U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),same) : {Bool.pick(Maybe<&2,U32>,_,Some{(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32)},strided(found,source_stride)) == strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride) : Maybe<&2,U32>}
            picked_index(U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),source,step,source_stride,first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside,apart,found,{==}) :
            {Bool.pick(Maybe<&2,U32>,Nat.is_le(U32.to_nat((U32.max(first,low) - low : U32)),digit) && Nat.is_lt(digit,Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)))),Some{(((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32) + U32.from_nat(Nat.sub(digit,U32.to_nat((U32.max(first,low) - low : U32)))) * (step * source_stride : U32) : U32)},strided(found,source_stride)) == strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride) : Maybe<&2,U32>}}),
        later)
    case False{}:
      +missed = Equal.trans(Bool,Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),False{},Equal.sym(Bool,U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),apart_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)),by_apart)
      +outside = Equal.trans(Bool,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),False{},cover_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside),uncovered(Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),digit)) && Nat.is_lt(Nat.add(U32.to_nat(low),digit),Nat.add(U32.to_nat(first),U32.to_nat(count))),first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside,missed,{==}))
      Equal.trans(Maybe<&2,U32>,last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,strided(found,source_stride)),last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,strided(Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),source_stride)),strided(segment_index(rest,(low + U32.from_nat(digit) : U32),Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found)),source_stride),
        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,acc => last_value(axis_boxes(rest,low,extent,source_stride,piece_stride),digit,strided(acc,source_stride)),found,Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found),
          {%Equal.sym(Bool,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),False{},outside) : {found == Bool.pick(Maybe<&2,U32>,_,Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found) : Maybe<&2,U32>} {==} : {found == Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found) : Maybe<&2,U32>}}),
        later)

# The box axes of an axis take the source offset of the last segment covering
# the position, times the source stride.
def axis_offset(segments: List<&2,Tasks.Segment>,+low: U32,+extent: U32,+source_stride: U32,+piece_stride: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+segments_in: SegmentsIn(segments,cap),+digit: Nat,+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},+found: Maybe<&2,U32>) ->
  {last_value(axis_boxes(segments,low,extent,source_stride,piece_stride),digit,strided(found,source_stride)) == strided(segment_index(segments,(low + U32.from_nat(digit) : U32),found),source_stride) : Maybe<&2,U32>}:
  match segments:
    case Nil{}: {==}
    case Con{Tasks.Segment{+first,+count,+source,+step},+rest}:
      Held.Both{+segment_room,+rest_in} = segments_in
      segment_step(U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),source,step,source_stride,piece_stride,rest,first,count,low,extent,cap,cap_ok,segment_room,piece_room,digit,inside,found,{==},
        axis_offset(rest,low,extent,source_stride,piece_stride,cap,cap_ok,piece_room,rest_in,digit,inside,Bool.pick(Maybe<&2,U32>,U32.is_le(first,(low + U32.from_nat(digit) : U32)) && U32.is_lt((low + U32.from_nat(digit) : U32),(first + count : U32)),Some{(source + ((low + U32.from_nat(digit) : U32) - first : U32) * step : U32)},found)))

def picked_boxes_in(picked: Bool,+source: U32,+step: U32,+source_stride: U32,+inner: List<&2,U32>,+rest: List<&2,Tasks.Segment>,+first: U32,+count: U32,+low: U32,+extent: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},
  +by_apart: {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == picked : Bool},
  +later: AxisBoxesIn(axis_boxes(rest,low,extent,source_stride,Grid.product(inner)),extent,inner)) ->
  AxisBoxesIn(Bool.pick(List<&2,BoxParts.BoxAxis>,picked,
    BoxParts.BoxAxis{Tasks.Cut{(U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32),(step * source_stride : U32),((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32),Grid.product(inner),((U32.max(first,low) - low : U32) * Grid.product(inner) : U32)},U32.to_nat((U32.max(first,low) - low : U32)),extent} <>
      axis_boxes(rest,low,extent,source_stride,Grid.product(inner)),axis_boxes(rest,low,extent,source_stride,Grid.product(inner))),extent,inner):
  match picked:
    case True{}:
      +apart = Equal.trans(Bool,Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),True{},Equal.sym(Bool,U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),apart_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)),by_apart)
      +low_max = NatExtrema.maximum_right(U32.to_nat(first),U32.to_nat(low))
      +max_min = Bounds.strict_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),apart)
      +low_min = Order.transitive(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),low_max,max_min)
      +restored_max = Order.restore_shift(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),low_max)
      +restored_min = Order.restore_shift(U32.to_nat(low),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),low_min)
      +span = Order.cancel_sum(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),
        Equal.trans(Nat,Nat.add(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),Addition.associative(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))))),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Equal.cong(Nat,Nat,sum => Nat.add(sum,Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),restored_max),
        Equal.trans(Nat,Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Order.restore_shift(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),max_min),
        Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),restored_min)))))
      +inside_piece = Equal.trans(Bool,Nat.is_le(Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),U32.to_nat(extent)),Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.add(U32.to_nat(low),U32.to_nat(extent))),True{},
        Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.is_le(Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),U32.to_nat(extent)),Order.le_offset(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),U32.to_nat(extent))),
        {%Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),restored_min) : {Nat.is_le(_,Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool} NatExtrema.minimum_right(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))) : {Nat.is_le(Nat.add(U32.to_nat(low),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low))),Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool}})
      Held.Both{{==},Held.Both{{==},Held.Both{
        Equal.cong(U32,U32,word => (word * Grid.product(inner) : U32),(U32.max(first,low) - low : U32),U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))),Equal.sym(U32,U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))),(U32.max(first,low) - low : U32),Arithmetic.native_roundtrip((U32.max(first,low) - low : U32)))),
        Held.Both{
          {%Equal.sym(Nat,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),count_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,apart)) : {Nat.is_lt(0n,_) == True{} : Bool} Order.positive_difference(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)),apart) : {Nat.is_lt(0n,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))) == True{} : Bool}},
          Held.Both{
            {%Equal.sym(Nat,U32.to_nat((U32.max(first,low) - low : U32)),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),offset_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)) : {Nat.is_le(Nat.add(_,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),U32.to_nat(extent)) == True{} : Bool}
              %Equal.sym(Nat,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),count_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,apart)) : {Nat.is_le(Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),_),U32.to_nat(extent)) == True{} : Bool}
              %Equal.sym(Nat,Nat.add(Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(low)),span) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
              inside_piece : {Nat.is_le(Nat.add(U32.to_nat((U32.max(first,low) - low : U32)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),U32.to_nat(extent)) == True{} : Bool}},
            later}}}}}
    case False{}: later

# The box axes of an axis are in the piece.
def axis_boxes_in(segments: List<&2,Tasks.Segment>,+low: U32,+extent: U32,+source_stride: U32,+inner: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+segments_in: SegmentsIn(segments,cap)) -> AxisBoxesIn(axis_boxes(segments,low,extent,source_stride,Grid.product(inner)),extent,inner):
  match segments:
    case Nil{}: Unit{}
    case Con{Tasks.Segment{+first,+count,+source,+step},+rest}:
      Held.Both{+segment_room,+rest_in} = segments_in
      picked_boxes_in(U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),source,step,source_stride,inner,rest,first,count,low,extent,cap,cap_ok,segment_room,piece_room,{==},axis_boxes_in(rest,low,extent,source_stride,inner,cap,cap_ok,piece_room,rest_in))

# Every axis of a placement has box axes in the piece.
def axes_boxed(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+axes: AxesIn(segments,coordinates,extents,source_strides,cap)) ->
  AxesBoxed(segments,coordinates,extents,source_strides,Tasks.strides_of(extents)):
  match segments:
    case Nil{}: axes
    case Con{+own,+segments}:
      match coordinates:
        case Nil{}: axes
        case Con{+low,+coordinates}:
          match extents:
            case Nil{}: axes
            case Con{+extent,+extents}:
              match source_strides:
                case Nil{}: axes
                case Con{+source_stride,+source_strides}:
                  Held.Both{+piece_room,Held.Both{+segments_in,+axes_more}} = axes
                  %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,extents}),Con{Grid.product(extents),Grid.strides(extents)},GridLaws.strides_con(extent,extents)) :
                    AxesBoxed(Con{own,segments},Con{low,coordinates},Con{extent,extents},Con{source_stride,source_strides},_)
                  Held.Both{axis_boxes_in(own,low,extent,source_stride,extents,cap,cap_ok,piece_room,segments_in),
                    axes_boxed(segments,coordinates,extents,source_strides,cap,cap_ok,axes_more)}

def ended_cell(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,+source_strides: List<&2,U32>,digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(Nil{},digits)) -> {Some{0} == segments_cell(segments,positions(coordinates,digits),source_strides) : Maybe<&2,U32>}:
  match segments:
    case Nil{}: {==}
    case Con{+own,+more}:
      match coordinates:
        case Nil{}: {==}
        case Con{+low,+rest}:
          match digits:
            case Nil{}: {==}
            case Con{+digit,+later}: Empty.absurd({Some{0} == segments_cell(Con{own,more},positions(Con{low,rest},Con{digit,later}),source_strides) : Maybe<&2,U32>},below)

# The box axes of every axis take the source cell of the segments covering
# the target coordinates.
def placed_cell(segments: List<&2,List<&2,Tasks.Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,digits: List<&2,Nat>,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+axes: AxesIn(segments,coordinates,extents,source_strides,cap),+below: Coordinates.DigitsBelow(extents,digits)) ->
  {placed_offset(segments,coordinates,extents,source_strides,Tasks.strides_of(extents),digits) == segments_cell(segments,positions(coordinates,digits),source_strides) : Maybe<&2,U32>}:
  match segments:
    case Nil{}: {==}
    case Con{+own,+segments}:
      match coordinates:
        case Nil{}: {==}
        case Con{+low,+coordinates}:
          match extents:
            case Nil{}: ended_cell(Con{own,segments},Con{low,coordinates},source_strides,digits,below)
            case Con{+extent,+extents}:
              match source_strides:
                case Nil{}:
                  match digits:
                    case Nil{}: Empty.absurd({placed_offset(Con{own,segments},Con{low,coordinates},Con{extent,extents},Nil{},Tasks.strides_of(Con{extent,extents}),Nil{}) == segments_cell(Con{own,segments},positions(Con{low,coordinates},Nil{}),Nil{}) : Maybe<&2,U32>},below)
                    case Con{+digit,+more}:
                      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,extents}),Con{Grid.product(extents),Grid.strides(extents)},GridLaws.strides_con(extent,extents)) :
                        {placed_offset(Con{own,segments},Con{low,coordinates},Con{extent,extents},Nil{},_,Con{digit,more}) ==
                          segments_cell(Con{own,segments},positions(Con{low,coordinates},Con{digit,more}),Nil{}) : Maybe<&2,U32>}
                      {==}
                case Con{+source_stride,+source_strides}:
                  match digits:
                    case Nil{}: Empty.absurd({placed_offset(Con{own,segments},Con{low,coordinates},Con{extent,extents},Con{source_stride,source_strides},Tasks.strides_of(Con{extent,extents}),Nil{}) == segments_cell(Con{own,segments},positions(Con{low,coordinates},Nil{}),Con{source_stride,source_strides}) : Maybe<&2,U32>},below)
                    case Con{+digit,+more}:
                      Held.Both{+piece_room,Held.Both{+segments_in,+axes_more}} = axes
                      KernelGemm.Both{+lower,+below_more} = below
                      +axis = last_value(axis_boxes(own,low,extent,source_stride,Grid.product(extents)),digit,None{})
                      +segment = strided(segment_index(own,(low + U32.from_nat(digit) : U32),None{}),source_stride)
                      +later = placed_offset(segments,coordinates,extents,source_strides,Tasks.strides_of(extents),more)
                      +cells = segments_cell(segments,positions(coordinates,more),source_strides)
                      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,extents}),Con{Grid.product(extents),Grid.strides(extents)},GridLaws.strides_con(extent,extents)) :
                        {placed_offset(Con{own,segments},Con{low,coordinates},Con{extent,extents},Con{source_stride,source_strides},_,Con{digit,more}) ==
                          segments_cell(Con{own,segments},positions(Con{low,coordinates},Con{digit,more}),Con{source_stride,source_strides}) : Maybe<&2,U32>}
                      Equal.trans(Maybe<&2,U32>,combined(axis,later),combined(segment,later),combined(segment,cells),
                        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,found => combined(found,later),axis,segment,
                          axis_offset(own,low,extent,source_stride,Grid.product(extents),cap,cap_ok,piece_room,segments_in,digit,lower,None{})),
                        Equal.cong(Maybe<&2,U32>,Maybe<&2,U32>,rest => combined(segment,rest),later,cells,
                          placed_cell(segments,coordinates,extents,source_strides,more,cap,cap_ok,axes_more,below_more)))

# ---- The parts of a placement ----

# The parts that place a source in a target piece take a piece cell from old
# to the source cell that the segments covering its target coordinates read,
# or leave it when an axis has none.
def placements_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+segments: List<&2,List<&2,Tasks.Segment>>,+source: List<&2,U32>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,+count: U32,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Grid.box_of(grid,cells,count))),U32.to_nat(cap)) == True{} : Bool},+query: U32,+query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(Grid.box_of(grid,cells,count))) == True{} : Bool},+old: Element,
  +axes: AxesIn(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),cap)) ->
  MoveParts.PartsAt(~Element,~Context,~value,context,Tasks.placements(segments,source,grid,cells,first,count),prefix_indices(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),query),query,old,placed_result(~Element,~Context,~value,context,segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old)):
  +whole = extents_twice(Grid.box_of(grid,cells,count),[])
  +boxed = axes_boxed(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),cap,cap_ok,axes)
  +held = prefixes_in(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}],[],boxed,Held.Both{Unit{},Held.Both{{==},Unit{}}})
  +room = {%Equal.sym(List<&2,U32>,reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]),Grid.box_of(grid,cells,count),whole) : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool} piece_room : {Nat.is_le(Nat.double(ShapeLaws.elements(reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]))),U32.to_nat(cap)) == True{} : Bool}}
  +inside = {%Equal.sym(List<&2,U32>,reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]),Grid.box_of(grid,cells,count),whole) : {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(_)) == True{} : Bool} query_in : {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]))) == True{} : Bool}}
  +below = Digits.coordinates_below(Grid.box_of(grid,cells,count),U32.to_nat(query),query_in)
  +Digits_below = below_reversed(Grid.box_of(grid,cells,count),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[],[],below,Unit{})
  +exact = Equal.trans(List<&2,Nat>,reversed_digits(reversed_digits(Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[]),[]),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),Targets.cell_coordinates(reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]),U32.to_nat(query)),
    digits_twice(Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[]),Equal.cong(List<&2,U32>,List<&2,Nat>,extents => Targets.cell_coordinates(extents,U32.to_nat(query)),Grid.box_of(grid,cells,count),reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]),Equal.sym(List<&2,U32>,reversed_extents(reversed_extents(Grid.box_of(grid,cells,count),[]),[]),Grid.box_of(grid,cells,count),whole)))
  +value_same = Equal.trans(Element,boxes_value(~Element,~Context,~value,context,placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),query,old),prefixes_value(~Element,~Context,~value,context,placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),reversed_digits(Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[]),Some{0},old),placed_result(~Element,~Context,~value,context,segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old),final_value(~Element,~Context,~value,context,placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),reversed_extents(Grid.box_of(grid,cells,count),[]),reversed_digits(Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[]),query,cap,cap_ok,room,exact,Digits_below,old,held),
        Equal.trans(Element,prefixes_value(~Element,~Context,~value,context,placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),reversed_digits(Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),[]),Some{0},old),prefixes_value(~Element,~Context,~value,context,[Prefix{[],0,0}],[],placed_offset(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old),placed_result(~Element,~Context,~value,context,segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old),placed_value(~Element,~Context,~value,context,segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}],[],Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),old,boxed,below),
        Equal.trans(Element,prefixes_value(~Element,~Context,~value,context,[Prefix{[],0,0}],[],placed_offset(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old),placed_result(~Element,~Context,~value,context,placed_offset(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old),placed_result(~Element,~Context,~value,context,segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old),first_result(~Element,~Context,~value,context,placed_offset(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),old),
        Equal.cong(Maybe<&2,U32>,Element,offset => placed_result(~Element,~Context,~value,context,offset,old),placed_offset(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),placed_cell(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query)),cap,cap_ok,axes,below)))))
  %Equal.sym(List<&2,Move.Part>,Tasks.placements(segments,source,grid,cells,first,count),prefix_parts(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}])),placements_prefixes(segments,source,grid,cells,first,count)) :
    MoveParts.PartsAt(~Element,~Context,~value,context,_,prefix_indices(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),query),query,old,placed_result(~Element,~Context,~value,context,segments_cell(segments,positions(Grid.coordinates(cells,first),Targets.cell_coordinates(Grid.box_of(grid,cells,count),U32.to_nat(query))),Tasks.strides_of(source)),old))
  %value_same : MoveParts.PartsAt(~Element,~Context,~value,context,prefix_parts(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}])),prefix_indices(placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),query),query,old,_)
  boxes_at(~Element,~Context,~value,context,placed_prefixes(segments,Grid.coordinates(cells,first),Grid.box_of(grid,cells,count),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(grid,cells,count)),[Prefix{[],0,0}]),reversed_extents(Grid.box_of(grid,cells,count),[]),cap,cap_ok,room,query,inside,old,held)
