# Tile writes read one lane at a time: lane l of the tile a program leaves
# at an address is what the program's lane-l values leave there.
import Base
import ../kernel_tile.bend as Tile
import ./kernel_tile_refinement.bend as TileLaws
import ./boolean_logic.bend as Boolean
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./write_program_every.bend as Every
import ../kernel_gemm.bend as KernelGemm

# Lane l of every write.
def lane_writes(+lane: Nat,writes: List<&2,Writes.Assignment<Tile.Tile8>>) -> List<&2,Writes.Assignment<F32>>:
  match writes:
    case []: []
    case Writes.Put{index,value} <> rest: Writes.Put{index,TileLaws.select_lane(lane,value)} <> lane_writes(lane,rest)

def picked_lane(+lane: Nat,condition: Bool,+chosen: Tile.Tile8,+other: Tile.Tile8) ->
  {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,condition,chosen,other)) == Bool.pick(F32,condition,TileLaws.select_lane(lane,chosen),TileLaws.select_lane(lane,other)) : F32}:
  match condition:
    case True{}: {==}
    case False{}: {==}

# Lane l of the tile a program selects at an address.
def selected_lane(+lane: Nat,writes: List<&2,Writes.Assignment<Tile.Tile8>>,+query: U32,+previous: Tile.Tile8) ->
  {TileLaws.select_lane(lane,Readback.selected(~Tile.Tile8,writes,query,previous)) ==
    Readback.selected(~F32,lane_writes(lane,writes),query,TileLaws.select_lane(lane,previous)) : F32}:
  match writes:
    case []: {==}
    case Writes.Put{+index,+value} <> +rest:
      +picked = Bool.pick(Tile.Tile8,U32.is_eq(index,query),value,previous)
      Equal.trans(F32,TileLaws.select_lane(lane,Readback.selected(~Tile.Tile8,rest,query,picked)),
        Readback.selected(~F32,lane_writes(lane,rest),query,TileLaws.select_lane(lane,picked)),
        Readback.selected(~F32,lane_writes(lane,rest),query,Bool.pick(F32,U32.is_eq(index,query),TileLaws.select_lane(lane,value),TileLaws.select_lane(lane,previous))),
        selected_lane(lane,rest,query,picked),
        Equal.cong(F32,F32,start => Readback.selected(~F32,lane_writes(lane,rest),query,start),TileLaws.select_lane(lane,picked),
          Bool.pick(F32,U32.is_eq(index,query),TileLaws.select_lane(lane,value),TileLaws.select_lane(lane,previous)),picked_lane(lane,U32.is_eq(index,query),value,previous)))

# A lane's writes reach the addresses the tile writes reach.
def contains_lane(+lane: Nat,+query: U32,writes: List<&2,Writes.Assignment<Tile.Tile8>>) ->
  {Readback.contains(~F32,query,lane_writes(lane,writes)) == Readback.contains(~Tile.Tile8,query,writes) : Bool}:
  match writes:
    case []: {==}
    case Writes.Put{+index,value} <> +rest:
      Equal.cong(Bool,Bool,later => U32.is_eq(index,query) || later,Readback.contains(~F32,query,lane_writes(lane,rest)),Readback.contains(~Tile.Tile8,query,rest),
        contains_lane(lane,query,rest))

# Lane l of every write at the address asked for is the wanted value, when
# every write states so of its own lane l.
def lane_matching(~Target: Data,~Holds: @+target: Target -> U32 -> Tile.Tile8 -> Data,~Context: Data,~target: @+context: Context -> Target,
  ~query: @+context: Context -> U32,~lane: @+context: Context -> Nat,~wanted: @+context: Context -> F32,
  ~matches: @+context: Context -> @+address: U32 -> @+value: Tile.Tile8 -> Holds(target(context),address,value) ->
    Boolean.When(U32.is_eq(address,query(context)),{TileLaws.select_lane(lane(context),value) == wanted(context) : F32}),
  +context: Context,writes: List<&2,Writes.Assignment<Tile.Tile8>>,holds: Every.Every(~Tile.Tile8,~Target,~Holds,target(context),writes)) ->
  Readback.Matching(~F32,query(context),wanted(context),lane_writes(lane(context),writes)):
  match writes:
    case []: Unit{}
    case Writes.Put{+address,+value} <> rest:
      KernelGemm.Both{here,later} = holds
      KernelGemm.Both{matches(context,address,value,here),lane_matching(~Target,~Holds,~Context,~target,~query,~lane,~wanted,~matches,context,rest,later)}
