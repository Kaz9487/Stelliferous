# The FP32 sum of the library: value i goes to lane i % 8, each lane keeps its
# rounded sum and the error of every addition (TwoSum), and the lanes merge
# in a fixed tree, (0,1),(2,3),(4,5),(6,7), then pairs of those.
import Base
import ./kernel_tile.bend as Tile
import ./traversal_loop.bend as Loop

# A rounded sum and the error it left out.
type Parts is Data:
  Parts{hi: F32,lo: F32}

# hi = a + b rounded, and lo with hi + lo = a + b exactly when nothing overflows.
def two_sum(+a: F32,+b: F32) -> Parts:
  +hi = (a + b : F32)
  +z = (hi - a : F32)
  Parts{hi,((a - (hi - z : F32) : F32) + (b - z : F32) : F32)}

# The eight lanes, lane 0 next. After n steps the lane of value n is in front:
# slot k holds lane (n + k) % 8, and phase counts the steps.
type Lanes is Data:
  Lanes{hi: Tile.Tile8,lo: Tile.Tile8,phase: U32}

def zero() -> Lanes:
  Lanes{Tile.filled(0.0),Tile.filled(0.0),0}

def turned(+h1: F32,+h2: F32,+h3: F32,+h4: F32,+h5: F32,+h6: F32,+h7: F32,l1: F32,l2: F32,l3: F32,l4: F32,l5: F32,l6: F32,l7: F32,l0: F32,
  +phase: U32,parts: Parts) -> Lanes:
  Parts{hi,error} = parts
  Lanes{Tile.Tile{h1,h2,h3,h4,h5,h6,h7,hi},Tile.Tile{l1,l2,l3,l4,l5,l6,l7,(l0 + error : F32)},(phase + 1 : U32)}

def stepped(+value: F32,lanes: Lanes) -> Lanes:
  Lanes{Tile.Tile{+h0,+h1,+h2,+h3,+h4,+h5,+h6,+h7},Tile.Tile{l0,l1,l2,l3,l4,l5,l6,l7},+phase} = lanes
  turned(h1,h2,h3,h4,h5,h6,h7,l1,l2,l3,l4,l5,l6,l7,l0,phase,two_sum(h0,value))

# The front lane takes the value, then the lanes turn by one.
def step(lanes: Lanes,value: F32) -> Lanes:
  stepped(value,lanes)

def merged(+lo: F32,parts: Parts) -> Parts:
  Parts{+hi,error} = parts
  two_sum(hi,(lo + error : F32))

def merge(left: Parts,right: Parts) -> Parts:
  Parts{+ah,al} = left
  Parts{+bh,bl} = right
  merged((al + bl : F32),two_sum(ah,bh))

def tree(p0: Parts,p1: Parts,p2: Parts,p3: Parts,p4: Parts,p5: Parts,p6: Parts,p7: Parts) -> Parts:
  merge(merge(merge(p0,p1),merge(p2,p3)),merge(merge(p4,p5),merge(p6,p7)))

# Lane j sits in slot (j - turns) % 8.
def in_order(turns: Nat,+p0: Parts,+p1: Parts,+p2: Parts,+p3: Parts,+p4: Parts,+p5: Parts,+p6: Parts,+p7: Parts) -> Parts:
  match turns:
    case 1n: tree(p7,p0,p1,p2,p3,p4,p5,p6)
    case 2n: tree(p6,p7,p0,p1,p2,p3,p4,p5)
    case 3n: tree(p5,p6,p7,p0,p1,p2,p3,p4)
    case 4n: tree(p4,p5,p6,p7,p0,p1,p2,p3)
    case 5n: tree(p3,p4,p5,p6,p7,p0,p1,p2)
    case 6n: tree(p2,p3,p4,p5,p6,p7,p0,p1)
    case 7n: tree(p1,p2,p3,p4,p5,p6,p7,p0)
    case turns: tree(p0,p1,p2,p3,p4,p5,p6,p7)

def rounded(parts: Parts) -> F32:
  Parts{hi,lo} = parts
  (0.0 + (hi + lo : F32) : F32)

# The lanes merged in their tree, before the rounding.
def parts(lanes: Lanes) -> Parts:
  Lanes{Tile.Tile{h0,h1,h2,h3,h4,h5,h6,h7},Tile.Tile{l0,l1,l2,l3,l4,l5,l6,l7},phase} = lanes
  in_order(U32.to_nat(U32.and(phase,7)),Parts{h0,l0},Parts{h1,l1},Parts{h2,l2},Parts{h3,l3},Parts{h4,l4},Parts{h5,l5},Parts{h6,l6},Parts{h7,l7})

# The sum of the values stepped into the lanes.
def total(lanes: Lanes) -> F32:
  rounded(parts(lanes))

# ---- Eight sums at once ----

# The parts of eight sums: lane j of each tile is the part of the sum of lane
# j.
type TileParts is Data:
  TileParts{hi: Tile.Tile8,lo: Tile.Tile8}

def tile_error(a: Tile.Tile8,b: Tile.Tile8,+hi: Tile.Tile8,+z: Tile.Tile8) -> Tile.Tile8:
  Tile.plus(Tile.minus(a,Tile.minus(hi,z)),Tile.minus(b,z))

def tile_rounded(a: Tile.Tile8,+copy: Tile.Tile8,b: Tile.Tile8,+hi: Tile.Tile8) -> TileParts:
  TileParts{hi,tile_error(a,b,hi,Tile.minus(hi,copy))}

# two_sum at every lane.
def tile_two_sum(+a: Tile.Tile8,+b: Tile.Tile8) -> TileParts:
  tile_rounded(a,a,b,Tile.plus(a,b))

# Slot k of eight sums at once holds the parts of the values whose index is k
# modulo 8: the value adds to the rounded sum and its error to the error.
def tile_slot_added(+lo: Tile.Tile8,parts: TileParts) -> TileParts:
  TileParts{hi,error} = parts
  TileParts{hi,Tile.plus(lo,error)}

def tile_slot_step(slot: TileParts,value: Tile.Tile8) -> TileParts:
  TileParts{+hi,lo} = slot
  tile_slot_added(lo,tile_two_sum(hi,value))

def tile_merged(+lo: Tile.Tile8,parts: TileParts) -> TileParts:
  TileParts{+hi,error} = parts
  tile_two_sum(hi,Tile.plus(lo,error))

# merge at every lane.
def tile_merge(left: TileParts,right: TileParts) -> TileParts:
  TileParts{+ah,al} = left
  TileParts{+bh,bl} = right
  tile_merged(Tile.plus(al,bl),tile_two_sum(ah,bh))

def tile_parts_zero() -> TileParts:
  +z = Tile.filled(0.0)
  TileParts{z,z}

# rounded at every lane.
def tile_rounded_total(parts: TileParts) -> Tile.Tile8:
  TileParts{hi,lo} = parts
  Tile.plus(Tile.filled(0.0),Tile.plus(hi,lo))
