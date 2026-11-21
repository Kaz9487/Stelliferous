# One lane of the slots of a row: the slots read from the phase on are the
# lanes of a sum of math_sum.bend at that lane, and the closed slots are its
# parts.
import Base
import ../kernel_tile.bend as Tile
import ../math_sum.bend as Sum
import ../tensor_operations.bend as Operations
import ../tensor_tiled_reduce.bend as TiledReduce
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_buffer_model.bend as Buffer
import ./kernel_tile_refinement.bend as TileLaws
import ./math_sum_slots.bend as Slots
import ./tensor_reduce_values.bend as Reduce
import ./tensor_tile_slots_values.bend as SlotsValues
import ../storage_buffer.bend as Storage

# ---- One lane of the parts ----

def parts_lane(+parts: Sum.TileParts,+lane: Nat) -> Sum.Parts:
  Sum.Parts{TileLaws.select_lane(lane,Slots.hi_of(parts)),TileLaws.select_lane(lane,Slots.lo_of(parts))}

def parts_hi(parts: Sum.Parts) -> F32:
  Sum.Parts{hi,lo} = parts
  hi

def parts_lo(parts: Sum.Parts) -> F32:
  Sum.Parts{hi,lo} = parts
  lo

# The value adds to the rounded sum of a slot and its error to the error.
def slot_added(+lo: F32,parts: Sum.Parts) -> Sum.Parts:
  Sum.Parts{hi,error} = parts
  Sum.Parts{hi,(lo + error : F32)}

def slot_step(parts: Sum.Parts,value: F32) -> Sum.Parts:
  Sum.Parts{+hi,lo} = parts
  slot_added(lo,Sum.two_sum(hi,value))

def slot_step_lane(+parts: Sum.TileParts,+value: Tile.Tile8,+lane: Nat) ->
  {parts_lane(Sum.tile_slot_step(parts,value),lane) == slot_step(parts_lane(parts,lane),TileLaws.select_lane(lane,value)) : Sum.Parts}:
  Sum.TileParts{+hi,+lo} = parts
  match hi lo value lane:
    case Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7} Tile.Tile{+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} Tile.Tile{+v0,+v1,+v2,+v3,+v4,+v5,+v6,+v7} 0n: {==}
    case Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7} Tile.Tile{+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} Tile.Tile{+v0,+v1,+v2,+v3,+v4,+v5,+v6,+v7} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def merge_tiles(+ah: Tile.Tile8,+al: Tile.Tile8,+bh: Tile.Tile8,+bl: Tile.Tile8,+lane: Nat) ->
  {parts_lane(Sum.tile_merge(Sum.TileParts{ah,al},Sum.TileParts{bh,bl}),lane) == Sum.merge(parts_lane(Sum.TileParts{ah,al},lane),parts_lane(Sum.TileParts{bh,bl},lane)) : Sum.Parts}:
  match ah al bh bl lane:
    case Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} Tile.Tile{+b0,+b1,+b2,+b3,+b4,+b5,+b6,+b7} Tile.Tile{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7} Tile.Tile{+d0,+d1,+d2,+d3,+d4,+d5,+d6,+d7} 0n: {==}
    case Tile.Tile{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} Tile.Tile{+b0,+b1,+b2,+b3,+b4,+b5,+b6,+b7} Tile.Tile{+c0,+c1,+c2,+c3,+c4,+c5,+c6,+c7} Tile.Tile{+d0,+d1,+d2,+d3,+d4,+d5,+d6,+d7} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def rounded_lane(+parts: Sum.TileParts,+lane: Nat) ->
  {TileLaws.select_lane(lane,Sum.tile_rounded_total(parts)) == Sum.rounded(parts_lane(parts,lane)) : F32}:
  Sum.TileParts{+hi,+lo} = parts
  match hi lo lane:
    case Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7} Tile.Tile{+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} 0n: {==}
    case Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7} Tile.Tile{+l0,+l1,+l2,+l3,+l4,+l5,+l6,+l7} 1n+next:
      match next:
        case 0n: {==}
        case 1n+next:
          match next:
            case 0n: {==}
            case 1n+next:
              match next:
                case 0n: {==}
                case 1n+next:
                  match next:
                    case 0n: {==}
                    case 1n+next:
                      match next:
                        case 0n: {==}
                        case 1n+next:
                          match next:
                            case 0n: {==}
                            case 1n+next: {==}

def merge_lane(+left: Sum.TileParts,+right: Sum.TileParts,+lane: Nat) ->
  {parts_lane(Sum.tile_merge(left,right),lane) == Sum.merge(parts_lane(left,lane),parts_lane(right,lane)) : Sum.Parts}:
  Sum.TileParts{+ah,+al} = left
  Sum.TileParts{+bh,+bl} = right
  merge_tiles(ah,al,bh,bl,lane)

def zero_lane(+lane: Nat) -> {parts_lane(Sum.tile_parts_zero(),lane) == TiledReduce.parts_zero() : Sum.Parts}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

# ---- Eight slots ----

type Eight is Data:
  Eight{s0: Sum.TileParts,s1: Sum.TileParts,s2: Sum.TileParts,s3: Sum.TileParts,s4: Sum.TileParts,s5: Sum.TileParts,s6: Sum.TileParts,s7: Sum.TileParts}

def tree_of(eight: Eight) -> Tree.Tree<Sum.TileParts>:
  Eight{s0,s1,s2,s3,s4,s5,s6,s7} = eight
  Tree.Node{Tree.Node{Tree.Node{Tree.Leaf{s0},Tree.Leaf{s1}},Tree.Node{Tree.Leaf{s2},Tree.Leaf{s3}}},
    Tree.Node{Tree.Node{Tree.Leaf{s4},Tree.Leaf{s5}},Tree.Node{Tree.Leaf{s6},Tree.Leaf{s7}}}}

def eight_of(+tree: Tree.Tree<Sum.TileParts>) -> Eight:
  Eight{Tree.read(Sum.TileParts,tree,0),Tree.read(Sum.TileParts,tree,1),Tree.read(Sum.TileParts,tree,2),Tree.read(Sum.TileParts,tree,3),Tree.read(Sum.TileParts,tree,4),Tree.read(Sum.TileParts,tree,5),Tree.read(Sum.TileParts,tree,6),Tree.read(Sum.TileParts,tree,7)}

def eight_tree(+eight: Eight) -> {eight_of(tree_of(eight)) == eight : Eight}:
  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
  {==}

def zeros() -> Eight:
  +zero = Sum.tile_parts_zero()
  Eight{zero,zero,zero,zero,zero,zero,zero,zero}

# The lanes of a sum at lane, its slots the parts of p0 to p7.
def lanes(+phase: U32,+lane: Nat,+p0: Sum.TileParts,+p1: Sum.TileParts,+p2: Sum.TileParts,+p3: Sum.TileParts,+p4: Sum.TileParts,+p5: Sum.TileParts,+p6: Sum.TileParts,+p7: Sum.TileParts) -> Sum.Lanes:
  Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(p0)),TileLaws.select_lane(lane,Slots.hi_of(p1)),TileLaws.select_lane(lane,Slots.hi_of(p2)),TileLaws.select_lane(lane,Slots.hi_of(p3)),TileLaws.select_lane(lane,Slots.hi_of(p4)),TileLaws.select_lane(lane,Slots.hi_of(p5)),TileLaws.select_lane(lane,Slots.hi_of(p6)),TileLaws.select_lane(lane,Slots.hi_of(p7))},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(p0)),TileLaws.select_lane(lane,Slots.lo_of(p1)),TileLaws.select_lane(lane,Slots.lo_of(p2)),TileLaws.select_lane(lane,Slots.lo_of(p3)),TileLaws.select_lane(lane,Slots.lo_of(p4)),TileLaws.select_lane(lane,Slots.lo_of(p5)),TileLaws.select_lane(lane,Slots.lo_of(p6)),TileLaws.select_lane(lane,Slots.lo_of(p7))},phase}

def lanes_in(eight: Eight,turn: Nat,+phase: U32,+lane: Nat) -> Sum.Lanes:
  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
  match turn:
    case 0n: lanes(phase,lane,s0,s1,s2,s3,s4,s5,s6,s7)
    case 1n+turn1:
      match turn1:
        case 0n: lanes(phase,lane,s1,s2,s3,s4,s5,s6,s7,s0)
        case 1n+turn2:
          match turn2:
            case 0n: lanes(phase,lane,s2,s3,s4,s5,s6,s7,s0,s1)
            case 1n+turn3:
              match turn3:
                case 0n: lanes(phase,lane,s3,s4,s5,s6,s7,s0,s1,s2)
                case 1n+turn4:
                  match turn4:
                    case 0n: lanes(phase,lane,s4,s5,s6,s7,s0,s1,s2,s3)
                    case 1n+turn5:
                      match turn5:
                        case 0n: lanes(phase,lane,s5,s6,s7,s0,s1,s2,s3,s4)
                        case 1n+turn6:
                          match turn6:
                            case 0n: lanes(phase,lane,s6,s7,s0,s1,s2,s3,s4,s5)
                            case 1n+turn7:
                              match turn7:
                                case 0n: lanes(phase,lane,s7,s0,s1,s2,s3,s4,s5,s6)
                                case 1n+turn8: lanes(phase,lane,s0,s1,s2,s3,s4,s5,s6,s7)

# The lanes of the eight slots from slot turn on.
def lanes_of(turn: Nat,+phase: U32,+lane: Nat,eight: Eight) -> Sum.Lanes:
  lanes_in(eight,turn,phase,lane)

def stepped_in(eight: Eight,turn: Nat,+value: Tile.Tile8) -> Eight:
  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
  match turn:
    case 0n: Eight{Sum.tile_slot_step(s0,value),s1,s2,s3,s4,s5,s6,s7}
    case 1n+turn1:
      match turn1:
        case 0n: Eight{s0,Sum.tile_slot_step(s1,value),s2,s3,s4,s5,s6,s7}
        case 1n+turn2:
          match turn2:
            case 0n: Eight{s0,s1,Sum.tile_slot_step(s2,value),s3,s4,s5,s6,s7}
            case 1n+turn3:
              match turn3:
                case 0n: Eight{s0,s1,s2,Sum.tile_slot_step(s3,value),s4,s5,s6,s7}
                case 1n+turn4:
                  match turn4:
                    case 0n: Eight{s0,s1,s2,s3,Sum.tile_slot_step(s4,value),s5,s6,s7}
                    case 1n+turn5:
                      match turn5:
                        case 0n: Eight{s0,s1,s2,s3,s4,Sum.tile_slot_step(s5,value),s6,s7}
                        case 1n+turn6:
                          match turn6:
                            case 0n: Eight{s0,s1,s2,s3,s4,s5,Sum.tile_slot_step(s6,value),s7}
                            case 1n+turn7:
                              match turn7:
                                case 0n: Eight{s0,s1,s2,s3,s4,s5,s6,Sum.tile_slot_step(s7,value)}
                                case 1n+turn8: Eight{Sum.tile_slot_step(s0,value),s1,s2,s3,s4,s5,s6,s7}

# Slot turn steps by the value.
def stepped(turn: Nat,eight: Eight,+value: Tile.Tile8) -> Eight:
  stepped_in(eight,turn,value)

def view_lanes(~Extra: Data,view: SlotsValues.SlotsView<Extra>,+lane: Nat) -> Sum.Lanes:
  SlotsValues.SlotsView{+slots,+phase,extra} = view
  lanes_of(U32.to_nat(U32.and(phase,7)),phase,lane,eight_of(slots))

# Writing slot turn of eight slots steps that slot.
def written(+turn: Nat,+eight: Eight,+value: Tile.Tile8) ->
  {Tree.write(Sum.TileParts,tree_of(eight),Slots.word(turn),Sum.tile_slot_step(Tree.read(Sum.TileParts,tree_of(eight),Slots.word(turn)),value)) ==
    tree_of(stepped(U32.to_nat(Slots.word(turn)),eight,value)) : Tree.Tree<Sum.TileParts>}:
  match turn:
    case 0n:
      Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
      {==}
    case 1n+turn1:
      match turn1:
        case 0n:
          Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
          {==}
        case 1n+turn2:
          match turn2:
            case 0n:
              Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
              {==}
            case 1n+turn3:
              match turn3:
                case 0n:
                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                  {==}
                case 1n+turn4:
                  match turn4:
                    case 0n:
                      Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                      {==}
                    case 1n+turn5:
                      match turn5:
                        case 0n:
                          Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                          {==}
                        case 1n+turn6:
                          match turn6:
                            case 0n:
                              Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                              {==}
                            case 1n+turn7:
                              match turn7:
                                case 0n:
                                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                                  {==}
                                case 1n+turn8:
                                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                                  {==}

def cell_written_at(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,+eight: Eight,+phase: U32,+extra: Extra,+cell: Tile.Tile8,+low: Slots.Low<phase>) ->
  {SlotsValues.cell_view(~Extra,~point,SlotsValues.SlotsView{tree_of(eight),phase,extra},cell) ==
    SlotsValues.SlotsView{tree_of(stepped(U32.to_nat(U32.and(phase,7)),eight,point(extra,cell))),(phase + 1 : U32),extra} : SlotsValues.SlotsView<Extra>}:
  Slots.Low{+turn,+now,+next} = low
  %Equal.sym(U32,U32.and(phase,7),Slots.word(turn),now) :
    {SlotsValues.SlotsView{Tree.write(Sum.TileParts,tree_of(eight),_,Sum.tile_slot_step(Tree.read(Sum.TileParts,tree_of(eight),_),point(extra,cell))),(phase + 1 : U32),extra} ==
      SlotsValues.SlotsView{tree_of(stepped(U32.to_nat(_),eight,point(extra,cell))),(phase + 1 : U32),extra} : SlotsValues.SlotsView<Extra>}
  Equal.cong(Tree.Tree<Sum.TileParts>,SlotsValues.SlotsView<Extra>,slots => SlotsValues.SlotsView{slots,(phase + 1 : U32),extra},
    Tree.write(Sum.TileParts,tree_of(eight),Slots.word(turn),Sum.tile_slot_step(Tree.read(Sum.TileParts,tree_of(eight),Slots.word(turn)),point(extra,cell))),
    tree_of(stepped(U32.to_nat(Slots.word(turn)),eight,point(extra,cell))),written(turn,eight,point(extra,cell)))

# A cell steps the slot of the phase of eight slots.
def cell_written(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,+eight: Eight,+phase: U32,+extra: Extra,+cell: Tile.Tile8) ->
  {SlotsValues.cell_view(~Extra,~point,SlotsValues.SlotsView{tree_of(eight),phase,extra},cell) ==
    SlotsValues.SlotsView{tree_of(stepped(U32.to_nat(U32.and(phase,7)),eight,point(extra,cell))),(phase + 1 : U32),extra} : SlotsValues.SlotsView<Extra>}:
  cell_written_at(~Extra,~point,eight,phase,extra,cell,Slots.low(phase))

# The lanes after a step of slot turn are the lanes of the sum stepped by the
# value at lane.
def lanes_cases(+turn: Nat,+phase: U32,+lane: Nat,+eight: Eight,+value: Tile.Tile8) ->
  {lanes_of(U32.to_nat(Slots.word(Slots.after(turn))),(phase + 1 : U32),lane,stepped(U32.to_nat(Slots.word(turn)),eight,value)) ==
    Sum.step(lanes_of(U32.to_nat(Slots.word(turn)),phase,lane,eight),TileLaws.select_lane(lane,value)) : Sum.Lanes}:
  match turn:
    case 0n:
      Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
      Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s0,value),lane),slot_step(parts_lane(s0,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s0,value,lane))
    case 1n+turn1:
      match turn1:
        case 0n:
          Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
          Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s1,value),lane),slot_step(parts_lane(s1,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s1,value,lane))
        case 1n+turn2:
          match turn2:
            case 0n:
              Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
              Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s2,value),lane),slot_step(parts_lane(s2,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s2,value,lane))
            case 1n+turn3:
              match turn3:
                case 0n:
                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                  Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s3,value),lane),slot_step(parts_lane(s3,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s3,value,lane))
                case 1n+turn4:
                  match turn4:
                    case 0n:
                      Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                      Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s4,value),lane),slot_step(parts_lane(s4,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s4,value,lane))
                    case 1n+turn5:
                      match turn5:
                        case 0n:
                          Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                          Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s5,value),lane),slot_step(parts_lane(s5,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s5,value,lane))
                        case 1n+turn6:
                          match turn6:
                            case 0n:
                              Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                              Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s7)),TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s7)),TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s6,value),lane),slot_step(parts_lane(s6,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s6,value,lane))
                            case 1n+turn7:
                              match turn7:
                                case 0n:
                                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                                  Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s0)),TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s0)),TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s7,value),lane),slot_step(parts_lane(s7,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s7,value,lane))
                                case 1n+turn8:
                                  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
                                  Equal.cong(Sum.Parts,Sum.Lanes,parts => Sum.Lanes{Tile.Tile{TileLaws.select_lane(lane,Slots.hi_of(s1)),TileLaws.select_lane(lane,Slots.hi_of(s2)),TileLaws.select_lane(lane,Slots.hi_of(s3)),TileLaws.select_lane(lane,Slots.hi_of(s4)),TileLaws.select_lane(lane,Slots.hi_of(s5)),TileLaws.select_lane(lane,Slots.hi_of(s6)),TileLaws.select_lane(lane,Slots.hi_of(s7)),parts_hi(parts)},Tile.Tile{TileLaws.select_lane(lane,Slots.lo_of(s1)),TileLaws.select_lane(lane,Slots.lo_of(s2)),TileLaws.select_lane(lane,Slots.lo_of(s3)),TileLaws.select_lane(lane,Slots.lo_of(s4)),TileLaws.select_lane(lane,Slots.lo_of(s5)),TileLaws.select_lane(lane,Slots.lo_of(s6)),TileLaws.select_lane(lane,Slots.lo_of(s7)),parts_lo(parts)},(phase + 1 : U32)},parts_lane(Sum.tile_slot_step(s0,value),lane),slot_step(parts_lane(s0,lane),TileLaws.select_lane(lane,value)),slot_step_lane(s0,value,lane))

# The lanes after a step of the slot of the phase are the lanes of the sum
# stepped by the value at lane.
def lanes_stepped(+turn: Nat,+phase: U32,+lane: Nat,+eight: Eight,+value: Tile.Tile8,
  +now: {U32.and(phase,7) == Slots.word(turn) : U32},+next: {U32.and((phase + 1 : U32),7) == Slots.word(Slots.after(turn)) : U32}) ->
  {lanes_of(U32.to_nat(U32.and((phase + 1 : U32),7)),(phase + 1 : U32),lane,stepped(U32.to_nat(U32.and(phase,7)),eight,value)) ==
    Sum.step(lanes_of(U32.to_nat(U32.and(phase,7)),phase,lane,eight),TileLaws.select_lane(lane,value)) : Sum.Lanes}:
  %Equal.sym(U32,U32.and(phase,7),Slots.word(turn),now) :
    {lanes_of(U32.to_nat(U32.and((phase + 1 : U32),7)),(phase + 1 : U32),lane,stepped(U32.to_nat(_),eight,value)) ==
      Sum.step(lanes_of(U32.to_nat(_),phase,lane,eight),TileLaws.select_lane(lane,value)) : Sum.Lanes}
  %Equal.sym(U32,U32.and((phase + 1 : U32),7),Slots.word(Slots.after(turn)),next) :
    {lanes_of(U32.to_nat(_),(phase + 1 : U32),lane,stepped(U32.to_nat(Slots.word(turn)),eight,value)) ==
      Sum.step(lanes_of(U32.to_nat(Slots.word(turn)),phase,lane,eight),TileLaws.select_lane(lane,value)) : Sum.Lanes}
  lanes_cases(turn,phase,lane,eight,value)

def lanes_next_at(~Extra: Data,+eight: Eight,+phase: U32,+extra: Extra,+value: Tile.Tile8,+lane: Nat,+low: Slots.Low<phase>) ->
  {view_lanes(~Extra,SlotsValues.SlotsView{tree_of(stepped(U32.to_nat(U32.and(phase,7)),eight,value)),(phase + 1 : U32),extra},lane) ==
    Sum.step(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane),TileLaws.select_lane(lane,value)) : Sum.Lanes}:
  Slots.Low{+turn,+now,+next} = low
  %Equal.sym(Eight,eight_of(tree_of(stepped(U32.to_nat(U32.and(phase,7)),eight,value))),stepped(U32.to_nat(U32.and(phase,7)),eight,value),
      eight_tree(stepped(U32.to_nat(U32.and(phase,7)),eight,value))) :
    {lanes_of(U32.to_nat(U32.and((phase + 1 : U32),7)),(phase + 1 : U32),lane,_) == Sum.step(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane),TileLaws.select_lane(lane,value)) : Sum.Lanes}
  %Equal.sym(Eight,eight_of(tree_of(eight)),eight,eight_tree(eight)) :
    {lanes_of(U32.to_nat(U32.and((phase + 1 : U32),7)),(phase + 1 : U32),lane,stepped(U32.to_nat(U32.and(phase,7)),eight,value)) ==
      Sum.step(lanes_of(U32.to_nat(U32.and(phase,7)),phase,lane,_),TileLaws.select_lane(lane,value)) : Sum.Lanes}
  lanes_stepped(turn,phase,lane,eight,value,now,next)

# The lanes after a cell are the lanes of the sum stepped by its lane.
def lanes_next(~Extra: Data,+eight: Eight,+phase: U32,+extra: Extra,+value: Tile.Tile8,+lane: Nat) ->
  {view_lanes(~Extra,SlotsValues.SlotsView{tree_of(stepped(U32.to_nat(U32.and(phase,7)),eight,value)),(phase + 1 : U32),extra},lane) ==
    Sum.step(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane),TileLaws.select_lane(lane,value)) : Sum.Lanes}:
  lanes_next_at(~Extra,eight,phase,extra,value,lane,Slots.low(phase))

# The parts of the lanes from the turn of the phase on merge the slots in
# order.
def ordered(+turn: Nat,+phase: U32,+lane: Nat,+s0: Sum.TileParts,+s1: Sum.TileParts,+s2: Sum.TileParts,+s3: Sum.TileParts,+s4: Sum.TileParts,+s5: Sum.TileParts,+s6: Sum.TileParts,+s7: Sum.TileParts,+same: {U32.to_nat(U32.and(phase,7)) == turn : Nat}) ->
  {Sum.parts(lanes_of(turn,phase,lane,Eight{s0,s1,s2,s3,s4,s5,s6,s7})) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}:
  match turn:
    case 0n:
      %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),0n,same) :
        {Sum.in_order(_,parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
      {==}
    case 1n+turn1:
      match turn1:
        case 0n:
          %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n,same) :
            {Sum.in_order(_,parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
          {==}
        case 1n+turn2:
          match turn2:
            case 0n:
              %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n,same) :
                {Sum.in_order(_,parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
              {==}
            case 1n+turn3:
              match turn3:
                case 0n:
                  %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n,same) :
                    {Sum.in_order(_,parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                  {==}
                case 1n+turn4:
                  match turn4:
                    case 0n:
                      %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n+1n,same) :
                        {Sum.in_order(_,parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                      {==}
                    case 1n+turn5:
                      match turn5:
                        case 0n:
                          %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n+1n+1n,same) :
                            {Sum.in_order(_,parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                          {==}
                        case 1n+turn6:
                          match turn6:
                            case 0n:
                              %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n+1n+1n+1n,same) :
                                {Sum.in_order(_,parts_lane(s6,lane),parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                              {==}
                            case 1n+turn7:
                              match turn7:
                                case 0n:
                                  %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n+1n+1n+1n+1n,same) :
                                    {Sum.in_order(_,parts_lane(s7,lane),parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                                  {==}
                                case 1n+turn8:
                                  %Equal.sym(Nat,U32.to_nat(U32.and(phase,7)),1n+1n+1n+1n+1n+1n+1n+1n+turn8,same) :
                                    {Sum.in_order(_,parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}
                                  {==}

# ---- The closed slots ----

def pair_read(+index: U32,+tree: Tree.Tree<Sum.TileParts>) ->
  {Operations.pair_at(index,Tree.pack(Sum.TileParts,tree)) ==
    (Tree.pack(Sum.TileParts,tree),Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32)))) : Array<Sum.TileParts> & Sum.TileParts}:
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Array.get(Sum.TileParts,Tree.pack(Sum.TileParts,tree),index),(Tree.pack(Sum.TileParts,tree),Tree.read(Sum.TileParts,tree,index)),
      StorageLaws.read(Sum.TileParts,tree,index)) :
    {Operations.pair_rest(index,_) ==
      (Tree.pack(Sum.TileParts,tree),Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32)))) : Array<Sum.TileParts> & Sum.TileParts}
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Array.get(Sum.TileParts,Tree.pack(Sum.TileParts,tree),(index + 1 : U32)),
      (Tree.pack(Sum.TileParts,tree),Tree.read(Sum.TileParts,tree,(index + 1 : U32))),StorageLaws.read(Sum.TileParts,tree,(index + 1 : U32))) :
    {Operations.merged_after(Tree.read(Sum.TileParts,tree,index),_) ==
      (Tree.pack(Sum.TileParts,tree),Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32)))) : Array<Sum.TileParts> & Sum.TileParts}
  {==}

def quad_merged(+index: U32,+tree: Tree.Tree<Sum.TileParts>) -> Sum.TileParts:
  Sum.tile_merge(Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32))),
    Sum.tile_merge(Tree.read(Sum.TileParts,tree,(index + 2 : U32)),Tree.read(Sum.TileParts,tree,((index + 2 : U32) + 1 : U32))))

def quad_read(+index: U32,+tree: Tree.Tree<Sum.TileParts>) ->
  {Operations.quad_at(index,Tree.pack(Sum.TileParts,tree)) == (Tree.pack(Sum.TileParts,tree),quad_merged(index,tree)) : Array<Sum.TileParts> & Sum.TileParts}:
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Operations.pair_at(index,Tree.pack(Sum.TileParts,tree)),
      (Tree.pack(Sum.TileParts,tree),Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32)))),pair_read(index,tree)) :
    {Operations.quad_rest(index,_) == (Tree.pack(Sum.TileParts,tree),quad_merged(index,tree)) : Array<Sum.TileParts> & Sum.TileParts}
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Operations.pair_at((index + 2 : U32),Tree.pack(Sum.TileParts,tree)),
      (Tree.pack(Sum.TileParts,tree),Sum.tile_merge(Tree.read(Sum.TileParts,tree,(index + 2 : U32)),Tree.read(Sum.TileParts,tree,((index + 2 : U32) + 1 : U32)))),pair_read((index + 2 : U32),tree)) :
    {Operations.merged_after(Sum.tile_merge(Tree.read(Sum.TileParts,tree,index),Tree.read(Sum.TileParts,tree,(index + 1 : U32))),_) ==
      (Tree.pack(Sum.TileParts,tree),quad_merged(index,tree)) : Array<Sum.TileParts> & Sum.TileParts}
  {==}

# The slots closed in place merge in their tree.
def closed_read(+tree: Tree.Tree<Sum.TileParts>) ->
  {Operations.eight_rest(Operations.quad_at(0,Tree.pack(Sum.TileParts,tree))) == Sum.tile_merge(quad_merged(0,tree),quad_merged(4,tree)) : Sum.TileParts}:
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Operations.quad_at(0,Tree.pack(Sum.TileParts,tree)),(Tree.pack(Sum.TileParts,tree),quad_merged(0,tree)),quad_read(0,tree)) :
    {Operations.eight_rest(_) == Sum.tile_merge(quad_merged(0,tree),quad_merged(4,tree)) : Sum.TileParts}
  %Equal.sym(Array<Sum.TileParts> & Sum.TileParts,Operations.quad_at(4,Tree.pack(Sum.TileParts,tree)),(Tree.pack(Sum.TileParts,tree),quad_merged(4,tree)),quad_read(4,tree)) :
    {Pair.snd(Array<Sum.TileParts>,Sum.TileParts,Operations.merged_after(quad_merged(0,tree),_)) == Sum.tile_merge(quad_merged(0,tree),quad_merged(4,tree)) : Sum.TileParts}
  {==}

# Lane lane of four slots merged in pairs, then the pairs.
def quad_lane(+a: Sum.TileParts,+b: Sum.TileParts,+c: Sum.TileParts,+d: Sum.TileParts,+lane: Nat) ->
  {parts_lane(Sum.tile_merge(Sum.tile_merge(a,b),Sum.tile_merge(c,d)),lane) ==
    Sum.merge(Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),Sum.merge(parts_lane(c,lane),parts_lane(d,lane))) : Sum.Parts}:
  Equal.trans(Sum.Parts,parts_lane(Sum.tile_merge(Sum.tile_merge(a,b),Sum.tile_merge(c,d)),lane),
    Sum.merge(parts_lane(Sum.tile_merge(a,b),lane),parts_lane(Sum.tile_merge(c,d),lane)),
    Sum.merge(Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),Sum.merge(parts_lane(c,lane),parts_lane(d,lane))),
    merge_lane(Sum.tile_merge(a,b),Sum.tile_merge(c,d),lane),
    Equal.trans(Sum.Parts,Sum.merge(parts_lane(Sum.tile_merge(a,b),lane),parts_lane(Sum.tile_merge(c,d),lane)),
      Sum.merge(Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),parts_lane(Sum.tile_merge(c,d),lane)),
      Sum.merge(Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),Sum.merge(parts_lane(c,lane),parts_lane(d,lane))),
      Equal.cong(Sum.Parts,Sum.Parts,left => Sum.merge(left,parts_lane(Sum.tile_merge(c,d),lane)),parts_lane(Sum.tile_merge(a,b),lane),
        Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),merge_lane(a,b,lane)),
      Equal.cong(Sum.Parts,Sum.Parts,right => Sum.merge(Sum.merge(parts_lane(a,lane),parts_lane(b,lane)),right),parts_lane(Sum.tile_merge(c,d),lane),
        Sum.merge(parts_lane(c,lane),parts_lane(d,lane)),merge_lane(c,d,lane))))

# Lane lane of the eight slots closed is the merge of their parts at lane.
def tree_lane(+s0: Sum.TileParts,+s1: Sum.TileParts,+s2: Sum.TileParts,+s3: Sum.TileParts,+s4: Sum.TileParts,+s5: Sum.TileParts,+s6: Sum.TileParts,+s7: Sum.TileParts,+lane: Nat) ->
  {parts_lane(Sum.tile_merge(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7))),lane) == Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)) : Sum.Parts}:
  Equal.trans(Sum.Parts,parts_lane(Sum.tile_merge(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7))),lane),Sum.merge(parts_lane(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),lane),parts_lane(Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane)),Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)),
    merge_lane(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane),
    Equal.trans(Sum.Parts,Sum.merge(parts_lane(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),lane),parts_lane(Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane)),Sum.merge(Sum.merge(Sum.merge(parts_lane(s0,lane),parts_lane(s1,lane)),Sum.merge(parts_lane(s2,lane),parts_lane(s3,lane))),parts_lane(Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane)),Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)),
      Equal.cong(Sum.Parts,Sum.Parts,left => Sum.merge(left,parts_lane(Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane)),parts_lane(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),lane),Sum.merge(Sum.merge(parts_lane(s0,lane),parts_lane(s1,lane)),Sum.merge(parts_lane(s2,lane),parts_lane(s3,lane))),quad_lane(s0,s1,s2,s3,lane)),
      Equal.cong(Sum.Parts,Sum.Parts,right => Sum.merge(Sum.merge(Sum.merge(parts_lane(s0,lane),parts_lane(s1,lane)),Sum.merge(parts_lane(s2,lane),parts_lane(s3,lane))),right),parts_lane(Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7)),lane),Sum.merge(Sum.merge(parts_lane(s4,lane),parts_lane(s5,lane)),Sum.merge(parts_lane(s6,lane),parts_lane(s7,lane))),quad_lane(s4,s5,s6,s7,lane))))

# Lane lane of closed eight slots is the parts of their lanes.
def closed_lane(~Extra: Data,+eight: Eight,+phase: U32,+extra: Extra,+lane: Nat) ->
  {parts_lane(SlotsValues.turn_view(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra}),lane) == Sum.parts(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane)) : Sum.Parts}:
  Eight{+s0,+s1,+s2,+s3,+s4,+s5,+s6,+s7} = eight
  Equal.trans(Sum.Parts,parts_lane(SlotsValues.turn_view(~Extra,SlotsValues.SlotsView{tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}),phase,extra}),lane),parts_lane(Sum.tile_merge(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7))),lane),
    Sum.parts(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}),phase,extra},lane)),
    Equal.cong(Sum.TileParts,Sum.Parts,parts => parts_lane(parts,lane),SlotsValues.turn_view(~Extra,SlotsValues.SlotsView{tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}),phase,extra}),Sum.tile_merge(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7))),closed_read(tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}))),
    Equal.trans(Sum.Parts,parts_lane(Sum.tile_merge(Sum.tile_merge(Sum.tile_merge(s0,s1),Sum.tile_merge(s2,s3)),Sum.tile_merge(Sum.tile_merge(s4,s5),Sum.tile_merge(s6,s7))),lane),Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)),Sum.parts(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}),phase,extra},lane)),
      tree_lane(s0,s1,s2,s3,s4,s5,s6,s7,lane),
      Equal.sym(Sum.Parts,Sum.parts(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(Eight{s0,s1,s2,s3,s4,s5,s6,s7}),phase,extra},lane)),Sum.tree(parts_lane(s0,lane),parts_lane(s1,lane),parts_lane(s2,lane),parts_lane(s3,lane),parts_lane(s4,lane),parts_lane(s5,lane),parts_lane(s6,lane),parts_lane(s7,lane)),
        ordered(U32.to_nat(U32.and(phase,7)),phase,lane,s0,s1,s2,s3,s4,s5,s6,s7,{==}))))

# ---- Folds of a line ----

# Lane lane of the cells a line reads, each taken by point.
type LanePoint<-Extra: Data,-Context: Data> is Data:
  LanePoint{lane: Nat,extra: Extra,context: Context}

def lane_point(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~Context: Data,~value: Context -> U32 -> Tile.Tile8,at: LanePoint<Extra,Context>,index: U32) -> F32:
  LanePoint{+lane,+extra,+context} = at
  TileLaws.select_lane(lane,point(extra,value(context,index)))

# Closed slots of a line of cells are the parts of the sum of its values at
# lane.
def closed_fold(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,count: Nat,+position: Nat,
  +eight: Eight,+phase: U32,+extra: Extra,+lane: Nat) ->
  {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,count,position,
      SlotsValues.SlotsView{tree_of(eight),phase,extra})),lane) ==
    Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},count,position,
      view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane))) : Sum.Parts}:
  match count:
    case 0n: closed_lane(~Extra,eight,phase,extra,lane)
    case 1n+ +rest:
      +cell = point(extra,value(context,U32.from_nat(position)))
      +next = stepped(U32.to_nat(U32.and(phase,7)),eight,cell)
      %Equal.sym(SlotsValues.SlotsView<Extra>,SlotsValues.cell_view(~Extra,~point,SlotsValues.SlotsView{tree_of(eight),phase,extra},value(context,U32.from_nat(position))),
          SlotsValues.SlotsView{tree_of(next),(phase + 1 : U32),extra},cell_written(~Extra,~point,eight,phase,extra,value(context,U32.from_nat(position)))) :
        {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,rest,1n+position,_)),lane) ==
          Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},rest,1n+position,
            Sum.step(view_lanes(~Extra,SlotsValues.SlotsView{tree_of(eight),phase,extra},lane),TileLaws.select_lane(lane,cell)))) : Sum.Parts}
      %lanes_next(~Extra,eight,phase,extra,cell,lane) :
        {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,rest,1n+position,
            SlotsValues.SlotsView{tree_of(next),(phase + 1 : U32),extra})),lane) ==
          Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},rest,1n+position,_)) : Sum.Parts}
      closed_fold(~Extra,~point,~Context,~value,context,rest,1n+position,next,(phase + 1 : U32),extra,lane)

# ---- The slots a row begins with ----

def begun(~Extra: Data,+extra: Extra) -> {SlotsValues.begin_view(~Extra,extra) == SlotsValues.SlotsView{tree_of(zeros()),0,extra} : SlotsValues.SlotsView<Extra>}:
  Equal.trans(SlotsValues.SlotsView<Extra>,SlotsValues.begin_view(~Extra,extra),
    SlotsValues.SlotsView{Buffer.content_tree(~Sum.TileParts,Buffer.contents(~Sum.TileParts,Storage.allocate(~Sum.TileParts,3n,Sum.tile_parts_zero(),8n))),0,extra},
    SlotsValues.SlotsView{tree_of(zeros()),0,extra},
    Equal.cong(Buffer.Content<Sum.TileParts>,SlotsValues.SlotsView<Extra>,content => SlotsValues.SlotsView{Buffer.content_tree(~Sum.TileParts,content),0,extra},
      Buffer.content(~Sum.TileParts,Buffer.remodel(~Sum.TileParts,Storage.allocate(~Sum.TileParts,3n,Sum.tile_parts_zero(),8n))),
      Buffer.contents(~Sum.TileParts,Storage.allocate(~Sum.TileParts,3n,Sum.tile_parts_zero(),8n)),
      Buffer.remodel_content(~Sum.TileParts,Storage.allocate(~Sum.TileParts,3n,Sum.tile_parts_zero(),8n))),
    Equal.cong(Tree.Tree<Sum.TileParts>,SlotsValues.SlotsView<Extra>,tree => SlotsValues.SlotsView{tree,0,extra},
      Tree.reflect(Sum.TileParts,Array.new(Sum.TileParts,3n,Sum.tile_parts_zero())),Tree.fresh(Sum.TileParts,3n,Sum.tile_parts_zero()),
      Equal.trans(Tree.Tree<Sum.TileParts>,Tree.reflect(Sum.TileParts,Array.new(Sum.TileParts,3n,Sum.tile_parts_zero())),
        Tree.reflect(Sum.TileParts,Tree.pack(Sum.TileParts,Tree.fresh(Sum.TileParts,3n,Sum.tile_parts_zero()))),Tree.fresh(Sum.TileParts,3n,Sum.tile_parts_zero()),
        Equal.cong(Array<Sum.TileParts>,Tree.Tree<Sum.TileParts>,storage => Tree.reflect(Sum.TileParts,storage),Array.new(Sum.TileParts,3n,Sum.tile_parts_zero()),
          Tree.pack(Sum.TileParts,Tree.fresh(Sum.TileParts,3n,Sum.tile_parts_zero())),StorageLaws.fresh(Sum.TileParts,3n,Sum.tile_parts_zero())),
        StorageLaws.reflect_packed(Sum.TileParts,Tree.fresh(Sum.TileParts,3n,Sum.tile_parts_zero())))))

def zero_lanes(~Extra: Data,+extra: Extra,+lane: Nat) -> {view_lanes(~Extra,SlotsValues.SlotsView{tree_of(zeros()),0,extra},lane) == Sum.zero() : Sum.Lanes}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g: {==}

# Lane lane of the closed slots of a line of cells is the parts of the sum of
# its values at lane.
def line_lane(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+count: Nat,+position: Nat,+extra: Extra,+lane: Nat) ->
  {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,count,position,
      SlotsValues.begin_view(~Extra,extra))),lane) ==
    Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},count,position,Sum.zero())) : Sum.Parts}:
  %Equal.sym(SlotsValues.SlotsView<Extra>,SlotsValues.begin_view(~Extra,extra),SlotsValues.SlotsView{tree_of(zeros()),0,extra},begun(~Extra,extra)) :
    {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,count,position,_)),lane) ==
      Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},count,position,Sum.zero())) : Sum.Parts}
  %zero_lanes(~Extra,extra,lane) :
    {parts_lane(SlotsValues.turn_view(~Extra,Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Context,~value,context,count,position,
        SlotsValues.SlotsView{tree_of(zeros()),0,extra})),lane) ==
      Sum.parts(Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~LanePoint<Extra,Context>,~lane_point(~Extra,~point,~Context,~value),LanePoint{lane,extra,context},count,position,_)) : Sum.Parts}
  closed_fold(~Extra,~point,~Context,~value,context,count,position,zeros(),0,extra,lane)
