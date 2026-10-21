# Bytes read from a file, as a list of words 0..255: positions, little-endian
# numbers, prefixes and text.
import Base

def at(bytes: List<&2,U32>,index: Nat) -> U32:
  match bytes index:
    case Nil{} index: 0
    case Con{byte,rest} 0n: byte
    case Con{byte,rest} 1n+before: at(rest,before)

# A number stored least significant byte first; a position past the end reads 0.
def u16_at(+bytes: List<&2,U32>,+index: Nat) -> U32:
  (at(bytes,index) + at(bytes,1n+index) * 256 : U32)

def u32_at(+bytes: List<&2,U32>,+index: Nat) -> U32:
  (u16_at(bytes,index) + u16_at(bytes,2n+index) * 65536 : U32)

def starts(pattern: List<&2,U32>,bytes: List<&2,U32>) -> Bool:
  match pattern bytes:
    case Nil{} bytes: True{}
    case Con{first,rest} Nil{}: False{}
    case Con{first,rest} Con{byte,more}: U32.is_eq(first,byte) && starts(rest,more)

def ends(+bytes: List<&2,U32>,+pattern: List<&2,U32>) -> Bool:
  starts(pattern,List.drop(&2,U32,bytes,Nat.sub(List.length(&2,U32,bytes),List.length(&2,U32,pattern))))

def same(left: List<&2,U32>,right: List<&2,U32>) -> Bool:
  match left right:
    case Nil{} Nil{}: True{}
    case Nil{} Con{byte,rest}: False{}
    case Con{byte,rest} Nil{}: False{}
    case Con{first,rest} Con{second,more}: U32.is_eq(first,second) && same(rest,more)

# The position of the last occurrence of the pattern, if any.
def last_start(bytes: List<&2,U32>,+pattern: List<&2,U32>,+index: U32,+best: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match bytes:
    case Nil{}: best
    case Con{+byte,+rest}:
      last_start(rest,pattern,(index + 1 : U32),Bool.pick(Maybe<&2,U32>,starts(pattern,byte <> rest),Some{index},best))

# ASCII text as bytes.
def codes(text: String) -> List<&2,U32>:
  match text:
    case SNil{}: []
    case SCon{head,tail}: Char.to_u32(head) <> codes(tail)

# UTF-8 bytes as text. A lead byte opens a code point of one to four bytes; each
# continuation byte adds six bits. A byte that fits no sequence is taken as is.

type Decoding is Data:
  Decoding{characters: List<&2,Char>,code: U32,remaining: U32}

def opened(four: Bool,three: Bool,two: Bool,+byte: U32,characters: List<&2,Char>) -> Decoding:
  match four three two:
    case True{} three two: Decoding{characters,(byte - 240 : U32),3}
    case False{} True{} two: Decoding{characters,(byte - 224 : U32),2}
    case False{} False{} True{}: Decoding{characters,(byte - 192 : U32),1}
    case False{} False{} False{}: Decoding{Char.from_u32(byte) <> characters,0,0}

def lead(+byte: U32,characters: List<&2,Char>) -> Decoding:
  opened(U32.is_le(240,byte),U32.is_le(224,byte),U32.is_le(192,byte),byte,characters)

def continued(last: Bool,+byte: U32,characters: List<&2,Char>,+code: U32,+remaining: U32) -> Decoding:
  match last:
    case True{}: Decoding{Char.from_u32((code * 64 + byte - 128 : U32)) <> characters,0,0}
    case False{}: Decoding{characters,(code * 64 + byte - 128 : U32),(remaining - 1 : U32)}

def stepped(fresh: Bool,+byte: U32,characters: List<&2,Char>,+code: U32,+remaining: U32) -> Decoding:
  match fresh:
    case True{}: lead(byte,characters)
    case False{}: continued(U32.is_eq(remaining,1),byte,characters,code,remaining)

def absorbed(+byte: U32,decoding: Decoding) -> Decoding:
  Decoding{characters,+code,+remaining} = decoding
  stepped(U32.is_zero(remaining),byte,characters,code,remaining)

def finished(decoding: Decoding) -> String:
  Decoding{characters,code,remaining} = decoding
  String.from_list(List.reverse(&2,Char,characters))

def decoded(bytes: List<&2,U32>,decoding: Decoding) -> String:
  match bytes:
    case Nil{}: finished(decoding)
    case Con{byte,rest}: decoded(rest,absorbed(byte,decoding))

def text(bytes: List<&2,U32>) -> String:
  decoded(bytes,Decoding{[],0,0})
