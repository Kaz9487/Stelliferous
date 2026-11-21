# The sums of math_sum.bend in their array: the sixteen cells, turned by the
# position of the next value, are the lanes of the sum with tiles in place of
# values, and lane j of those is the state of the sum of lane j of the values.
import Base
import ../kernel_tile.bend as Tile
import ../math_sum.bend as Sum
import ../tensor_tiled_reduce.bend as TiledReduce
import ./word_encoding.bend as Words

# ---- The position of the next value ----

def word(turn: Nat) -> U32:
  match turn:
    case 1n: 1
    case 2n: 2
    case 3n: 3
    case 4n: 4
    case 5n: 5
    case 6n: 6
    case 7n: 7
    case turn: 0

def after(turn: Nat) -> Nat:
  match turn:
    case 1n: 2n
    case 2n: 3n
    case 3n: 4n
    case 4n: 5n
    case 5n: 6n
    case 6n: 7n
    case 7n: 0n
    case turn: 1n

def masked(+a: Bool,+b: Bool,+c: Bool,+rest: Word(29n)) ->
  {U32{WCon{a,WCon{b,WCon{c,Word.and(29n,rest,Word.zero(29n))}}}} == U32{WCon{a,WCon{b,WCon{c,Word.zero(29n)}}}} : U32}:
  Equal.cong(Word(29n),U32,tail => U32{WCon{a,WCon{b,WCon{c,tail}}}},Word.and(29n,rest,Word.zero(29n)),Word.zero(29n),Words.and_zero(29n,rest))

# The turn of a position and of the next one.
type Low<-phase: U32> is Data:
  Low{turn: Nat,now: {U32.and(phase,7) == word(turn) : U32},next: {U32.and((phase + 1 : U32),7) == word(after(turn)) : U32}}

def low(phase: U32) -> Low<phase>:
  match phase:
    case U32{WCon{False{},WCon{False{},WCon{False{},+rest}}}}:
      Low{0n,masked(False{},False{},False{},rest),masked(True{},False{},False{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{True{},WCon{False{},WCon{False{},+rest}}}}:
      Low{1n,masked(True{},False{},False{},rest),masked(False{},True{},False{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{False{},WCon{True{},WCon{False{},+rest}}}}:
      Low{2n,masked(False{},True{},False{},rest),masked(True{},True{},False{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{True{},WCon{True{},WCon{False{},+rest}}}}:
      Low{3n,masked(True{},True{},False{},rest),masked(False{},False{},True{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{False{},WCon{False{},WCon{True{},+rest}}}}:
      Low{4n,masked(False{},False{},True{},rest),masked(True{},False{},True{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{True{},WCon{False{},WCon{True{},+rest}}}}:
      Low{5n,masked(True{},False{},True{},rest),masked(False{},True{},True{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{False{},WCon{True{},WCon{True{},+rest}}}}:
      Low{6n,masked(False{},True{},True{},rest),masked(True{},True{},True{},Word.adc(29n,rest,Word.zero(29n),False{},False{}))}
    case U32{WCon{True{},WCon{True{},WCon{True{},+rest}}}}:
      Low{7n,masked(True{},True{},True{},rest),masked(False{},False{},False{},Word.adc(29n,rest,Word.zero(29n),False{},True{}))}

# The position count values later.
def ahead(count: Nat,+word: U32) -> U32:
  match count:
    case 0n: word
    case 1n+rest: ahead(rest,(word + 1 : U32))

def carried(+carry: Bool,+rest: Word(29n)) -> Word(29n):
  Word.adc(29n,rest,Word.zero(29n),False{},carry)

# The bits above the low three, eight positions later.
def eight_later(+rest: Word(29n)) -> Word(29n):
  carried(True{},carried(False{},carried(False{},carried(False{},carried(False{},carried(False{},carried(False{},carried(False{},rest))))))))

def hi_of(parts: Sum.TileParts) -> Tile.Tile8:
  Sum.TileParts{hi,lo} = parts
  hi

def lo_of(parts: Sum.TileParts) -> Tile.Tile8:
  Sum.TileParts{hi,lo} = parts
  lo
