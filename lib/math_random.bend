# Random numbers by counter: the word of an element depends only on the seed,
# the stream and its index, so results do not depend on the worker count. Each
# word is the first output of Threefry-2x32 with 20 rounds (Salmon et al. 2011)
# for the key (seed, stream) and the counter (index, 0).
import Base
import ./math_fp32.bend as Fp32

type Key is Data:
  Seeded{seed: U32,stream: U32}

# Two words of the cipher state.
type Block is Data:
  Block{x: U32,y: U32}

def rotated(+word: U32,+left: Nat,+right: Nat) -> U32:
  U32.or(U32.shln(word,left),U32.shrn(word,right))

def round(block: Block,+left: Nat,+right: Nat) -> Block:
  Block{+x,+y} = block
  +sum = (x + y : U32)
  Block{sum,U32.xor(rotated(y,left,right),sum)}

def keyed(block: Block,+first: U32,+second: U32) -> Block:
  Block{+x,+y} = block
  Block{(x + first : U32),(y + second : U32)}

# The four rounds of the even and of the odd schedule.
def even_rounds(block: Block) -> Block:
  round(round(round(round(block,13n,19n),15n,17n),26n,6n),6n,26n)

def odd_rounds(block: Block) -> Block:
  round(round(round(round(block,17n,15n),29n,3n),16n,16n),24n,8n)

# The first word of the cipher of the counter under the key words.
def enciphered(+first: U32,+second: U32,+third: U32,+counter: U32,+high: U32) -> Block:
  keyed(even_rounds(keyed(odd_rounds(keyed(even_rounds(keyed(odd_rounds(keyed(even_rounds(keyed(Block{counter,high},first,second)),
    second,(third + 1 : U32))),third,(first + 2 : U32))),first,(second + 3 : U32))),second,(third + 4 : U32))),third,(first + 5 : U32))

def block(+seed: U32,+stream: U32,+index: U32,+high: U32) -> Block:
  enciphered(seed,stream,U32.xor(U32.xor(seed,stream),466688986),index,high)

def first(block: Block) -> U32:
  Block{x,y} = block
  x

def word(+key: Key,+index: U32) -> U32:
  Seeded{+seed,+stream} = key
  first(block(seed,stream,index,0))

# The top 24 bits of the word scaled to [0, 1), exactly.
def uniform(+key: Key,+index: U32) -> F32:
  (U32.to_f32(U32.shrn(word(key,index),8n)) * 0.00000005960464477539063 : F32)

# The top 32 bits of the 64-bit product of two words.
def high_product(+left: U32,+right: U32) -> U32:
  +a = U32.shrn(left,16n)
  +b = U32.and(left,65535)
  +c = U32.shrn(right,16n)
  +d = U32.and(right,65535)
  +middle = (a * d : U32)
  +cross = (b * c : U32)
  +carry = U32.shrn((U32.and(middle,65535) + U32.and(cross,65535) + U32.shrn((b * d : U32),16n) : U32),16n)
  (a * c + U32.shrn(middle,16n) + U32.shrn(cross,16n) + carry : U32)

# An integer below the bound, by the top of the product of the word and the
# bound, as an FP32.
def below(+bound: U32,+key: Key,+index: U32) -> F32:
  U32.to_f32(high_product(word(key,index),bound))

# The counter is carried in the element's bits, which are only moved.
def counter(index: U32,value: F32) -> F32:
  Fp32.from_bits(index)

def uniform_of_counter(key: Key,value: F32) -> F32:
  +key = key
  uniform(key,Fp32.to_bits(value))

type Bounded is Data:
  Bounded{bound: U32,key: Key}

def below_of_counter(bounded: Bounded,value: F32) -> F32:
  Bounded{+bound,+key} = bounded
  below(bound,key,Fp32.to_bits(value))
