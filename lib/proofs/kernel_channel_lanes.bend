# The lane of a channel in the cell of its group: the bits the packer
# computes from the channel select lane channel % 8.
import Base
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ./kernel_tile_refinement.bend as TileLaws
import ./u32_division.bend as WordDivision
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division

# The bits of the lane of a channel, as the packer computes them.
type Bits is Data:
  Bits{half: Bool,low: Bool,first: Bool}

def bits(+channel: U32) -> Bits:
  +lane = (channel % 8 : U32)
  Bits{U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32))}

def lane_dispatch(half: Bool,low: Bool,first: Bool,tile: Tile.Tile8) -> F32:
  match half low first:
    case True{} True{} True{}: Tile.lane0(tile)
    case True{} True{} False{}: Tile.lane1(tile)
    case True{} False{} True{}: Tile.lane2(tile)
    case True{} False{} False{}: Tile.lane3(tile)
    case False{} True{} True{}: Tile.lane4(tile)
    case False{} True{} False{}: Tile.lane5(tile)
    case False{} False{} True{}: Tile.lane6(tile)
    case False{} False{} False{}: Tile.lane7(tile)

def lane_at(lanes: Bits,tile: Tile.Tile8) -> F32:
  Bits{half,low,first} = lanes
  lane_dispatch(half,low,first,tile)

# The values of the channels of a plane fit one storage of a machine, and g is
# the number of groups of the channels.
type Fits<-channels: U32,-h: U32,-w: U32,-g: U32> is Data:
  Fits{limits: Machine.Limits,grouped: {g == Grid.groups(channels) : U32},
    plane_exact: {Nat.mul(U32.to_nat(h),U32.to_nat(w)) == U32.to_nat((h * w : U32)) : Nat},
    total_exact: {Nat.mul(U32.to_nat(channels),U32.to_nat((h * w : U32))) == U32.to_nat((channels * (h * w) : U32)) : Nat},
    bounded: {U32.is_le((channels * (h * w) : U32),Machine.capacity(limits)) == True{} : Bool}}

def lane_of_word(+lane: U32,tile: Tile.Tile8) -> F32:
  lane_dispatch(U32.is_lt(lane,4),U32.is_lt((lane % 4 : U32),2),U32.is_zero((lane % 2 : U32)),tile)

def LaneNumber(+lane: Nat,+tile: Tile.Tile8) -> Type:
  {TileLaws.select_lane(lane,tile) == lane_of_word(U32.from_nat(lane),tile) : F32}

# A lane below eight selects the tile field its bits name.
def lane_number(+lane: Nat,+tile: Tile.Tile8,below: {Nat.is_lt(lane,8n) == True{} : Bool}) -> LaneNumber(lane,tile):
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+ +g:
                              match g:
                                case 0n: {==}
                                case 1n+ +r: Empty.absurd(LaneNumber(Nat.add(8n,r),tile),WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(r,0n),True{},Equal.sym(Bool,Nat.is_lt(r,0n),False{},Division.lt_zero(r)),below)))

# The lane the bits of a channel name is lane channel % 8.
def lane_bits(+c: U32,+tile: Tile.Tile8) -> {lane_at(bits(c),tile) == TileLaws.select_lane(Nat.mod(U32.to_nat(c),8n),tile) : F32}:
  +lane = (c % 8 : U32)
  +number = U32.to_nat(lane)
  +exact = WordDivision.modulo(c,8,16,{==},{==},{==})
  +below = {%Equal.sym(Nat,number,Nat.mod(U32.to_nat(c),8n),exact) : {Nat.is_lt(_,8n) == True{} : Bool} Radix.remainder_inside(c,8n,{==}) : {Nat.is_lt(number,8n) == True{} : Bool}}
  %exact : {lane_at(bits(c),tile) == TileLaws.select_lane(_,tile) : F32}
  Equal.sym(F32,TileLaws.select_lane(number,tile),lane_at(bits(c),tile),
    %Arithmetic.native_roundtrip(lane) : {TileLaws.select_lane(number,tile) == lane_of_word(_,tile) : F32}
    lane_number(number,tile,below))
