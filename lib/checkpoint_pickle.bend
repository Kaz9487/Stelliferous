# A pickle, the object graph torch.save writes, replayed opcode by opcode into
# values: the stack machine of protocol 2 with the string, global and memo
# opcodes of protocol 4. No code runs: a class is a Global, an instance keeps
# its class, arguments and state, and a call keeps its callable and arguments.
import Base
import ./storage_bytes.bend as Bytes
import ./storage_buffer.bend as Buffers

type Value is Data:
  Absent{}
  Flag{value: Bool}
  Integer{word: U32}
  Float{high: U32,low: U32}
  Text{text: String}
  Sequence{items: List<&2,Value>}
  Items{items: List<&2,Value>}
  Dict{entries: List<&2,Value>}
  Global{module: String,name: String}
  Object{class: Value,arguments: Value,state: Value}
  Reduced{callable: Value,arguments: Value}
  Persistent{id: Value}
  Mark{}

type Parsed is Data:
  Parsed{value: Value}
  Rejected{detail: String}

# The stack, the memo with its capacity, and the index the next memoize takes.
type Machine is Type:
  Machine{stack: List<&2,Value>,memo: Array<Value>,capacity: U32,next: U32}

type Step is Type:
  Running{bytes: List<&2,U32>,machine: Machine}
  Completed{value: Value}
  Halted{detail: String}

type Split is Data:
  Split{items: List<&2,Value>,rest: List<&2,Value>}
  Unmarked{}

type Line is Data:
  Line{text: List<&2,U32>,rest: List<&2,U32>}

# Dictionaries: entries are two-item sequences; assignment replaces an existing key.

def same_key(left: Value,right: Value) -> Bool:
  match left right:
    case Text{left} Text{right}: String.eq(left,right)
    case Integer{left} Integer{right}: U32.is_eq(left,right)
    case left right: False{}

def key_of(entry: Value) -> Value:
  match entry:
    case Sequence{Con{key,others}}: key
    case entry: Absent{}

def assign(entries: List<&2,Value>,+key: Value,+value: Value) -> List<&2,Value>:
  match entries:
    case Nil{}: [Sequence{[key,value]}]
    case Con{+entry,+rest}: Bool.pick(List<&2,Value>,same_key(key_of(entry),key),Sequence{[key,value]} <> rest,entry <> assign(rest,key,value))

def assign_pairs(pairs: List<&2,Value>,entries: List<&2,Value>) -> List<&2,Value>:
  match pairs:
    case Con{key,Con{value,more}}: assign_pairs(more,assign(entries,key,value))
    case pairs: entries

# The value under a text key, or Absent.
def get(entries: List<&2,Value>,+key: String) -> Value:
  match entries:
    case Nil{}: Absent{}
    case Con{Sequence{Con{Text{+name},Con{+value,others}}},rest}: Bool.pick(Value,String.eq(name,key),value,get(rest,key))
    case Con{entry,rest}: get(rest,key)

# The items above the mark, oldest first, and the stack below it.
def split(stack: List<&2,Value>,items: List<&2,Value>) -> Split:
  match stack:
    case Nil{}: Unmarked{}
    case Con{Mark{},rest}: Split{items,rest}
    case Con{top,rest}: split(rest,top <> items)

def line(bytes: List<&2,U32>,ended: Bool,+text: List<&2,U32>) -> Line:
  match bytes ended:
    case bytes True{}: Line{List.reverse(&2,U32,text),bytes}
    case Nil{} False{}: Line{List.reverse(&2,U32,text),[]}
    case Con{+byte,rest} False{}: line(rest,U32.is_eq(byte,10),Bool.pick(List<&2,U32>,U32.is_eq(byte,10),text,byte <> text))

# Little-endian integers of one to four bytes, sign-extended as the LONG1 opcode means them.
def sign_extended(+length: U32,+word: U32) -> Value:
  match length:
    case 1: Integer{Bool.pick(U32,U32.is_le(128,word),(word + 4294967040 : U32),word)}
    case 2: Integer{Bool.pick(U32,U32.is_le(32768,word),(word + 4294901760 : U32),word)}
    case 3: Integer{Bool.pick(U32,U32.is_le(8388608,word),(word + 4278190080 : U32),word)}
    case _: Integer{word}

def little_endian(bytes: List<&2,U32>,+weight: U32) -> U32:
  match bytes:
    case Nil{}: 0
    case Con{byte,rest}: (byte * weight + little_endian(rest,(weight * 256 : U32)) : U32)

# Opcodes

def push(value: Value,machine: Machine) -> Machine:
  Machine{stack,memo,capacity,next} = machine
  Machine{value <> stack,memo,capacity,next}

def pushed(bytes: List<&2,U32>,value: Value,machine: Machine) -> Step:
  Running{bytes,push(value,machine)}

def stored(bytes: List<&2,U32>,fits: Bool,+index: U32,+top: Value,stack: List<&2,Value>,memo: Array<Value>,+capacity: U32,+next: U32) -> Step:
  match fits:
    case False{}: Halted{"memo index " ++ U32.show(index) ++ " exceeds the pickle"}
    case True{}: Running{bytes,Machine{top <> stack,Array.set(Value,memo,index,top),capacity,U32.max(next,(index + 1 : U32))}}

# BINPUT, LONG_BINPUT and MEMOIZE keep the top of the stack under an index.
def remember(bytes: List<&2,U32>,+index: U32,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  match stack:
    case Nil{}: Halted{"PUT on an empty stack"}
    case Con{+top,rest}: stored(bytes,U32.is_lt(index,capacity),index,top,rest,memo,capacity,next)

def recalled(bytes: List<&2,U32>,pair: Array<Value> & Value,stack: List<&2,Value>,+capacity: U32,+next: U32) -> Step:
  (memo,value) = pair
  Running{bytes,Machine{value <> stack,memo,capacity,next}}

def recall_within(bytes: List<&2,U32>,fits: Bool,+index: U32,stack: List<&2,Value>,memo: Array<Value>,+capacity: U32,+next: U32) -> Step:
  match fits:
    case False{}: Halted{"memo index " ++ U32.show(index) ++ " exceeds the pickle"}
    case True{}: recalled(bytes,Array.get(Value,memo,index),stack,capacity,next)

# BINGET and LONG_BINGET push the value under an index.
def recall(bytes: List<&2,U32>,+index: U32,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  recall_within(bytes,U32.is_lt(index,capacity),index,stack,memo,capacity,next)

def memoize(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  remember(bytes,next,Machine{stack,memo,capacity,next})

def put_short(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{index,rest}: remember(rest,index,machine)
    case bytes: Halted{"BINPUT without its index"}

def put_long(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}: remember(rest,little_endian([first,second,third,fourth],1),machine)
    case bytes: Halted{"LONG_BINPUT without its index"}

def get_short(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{index,rest}: recall(rest,index,machine)
    case bytes: Halted{"BINGET without its index"}

def get_long(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}: recall(rest,little_endian([first,second,third,fourth],1),machine)
    case bytes: Halted{"LONG_BINGET without its index"}

# Text: a length, then that many UTF-8 bytes.
def text_of(+length: U32,+bytes: List<&2,U32>,machine: Machine) -> Step:
  Running{List.drop(&2,U32,bytes,U32.to_nat(length)),push(Text{Bytes.text(List.take(&2,U32,bytes,U32.to_nat(length)))},machine)}

def text_short(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{length,rest}: text_of(length,rest,machine)
    case bytes: Halted{"a short string without its length"}

def text_long(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}: text_of(little_endian([first,second,third,fourth],1),rest,machine)
    case bytes: Halted{"a string without its length"}

# GLOBAL: a module and a name, each ended by a newline.
def global_named(second: Line,+module: List<&2,U32>,machine: Machine) -> Step:
  Line{name,rest} = second
  Running{rest,push(Global{Bytes.text(module),Bytes.text(name)},machine)}

def global_module(first: Line,machine: Machine) -> Step:
  Line{module,rest} = first
  global_named(line(rest,False{},[]),module,machine)

def global(bytes: List<&2,U32>,machine: Machine) -> Step:
  global_module(line(bytes,False{},[]),machine)

# STACK_GLOBAL takes the name and the module from the stack.
def stack_global(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{Text{name},Con{Text{module},rest}}: Running{bytes,Machine{Global{module,name} <> rest,memo,capacity,next}}
    case stack: Halted{"STACK_GLOBAL without two strings"}

# Numbers

def integer_short(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{byte,rest}: pushed(rest,Integer{byte},machine)
    case bytes: Halted{"BININT1 without its byte"}

def integer_two(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{first,Con{second,rest}}: pushed(rest,Integer{little_endian([first,second],1)},machine)
    case bytes: Halted{"BININT2 without its bytes"}

def integer_four(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}: pushed(rest,Integer{little_endian([first,second,third,fourth],1)},machine)
    case bytes: Halted{"BININT without its bytes"}

def long_sized(narrow: Bool,+length: U32,+bytes: List<&2,U32>,machine: Machine) -> Step:
  match narrow:
    case False{}: Halted{"an integer wider than 32 bits"}
    case True{}: Running{List.drop(&2,U32,bytes,U32.to_nat(length)),push(sign_extended(length,little_endian(List.take(&2,U32,bytes,U32.to_nat(length)),1)),machine)}

# LONG1: a byte count, then the two's complement bytes.
def long_one(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{+length,rest}: long_sized(U32.is_le(length,4),length,rest,machine)
    case bytes: Halted{"LONG1 without its length"}

# BINFLOAT: eight bytes, most significant first.
def float(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Con{a,Con{b,Con{c,Con{d,Con{e,Con{f,Con{g,Con{h,rest}}}}}}}}:
      pushed(rest,Float{little_endian([d,c,b,a],1),little_endian([h,g,f,e],1)},machine)
    case bytes: Halted{"BINFLOAT without its bytes"}

# Containers

def tuple_of(parts: Split,bytes: List<&2,U32>,memo: Array<Value>,+capacity: U32,+next: U32) -> Step:
  match parts:
    case Unmarked{}: Halted{"TUPLE without a mark"}
    case Split{items,rest}: Running{bytes,Machine{Sequence{items} <> rest,memo,capacity,next}}

def tuple(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  tuple_of(split(stack,[]),bytes,memo,capacity,next)

def tuple_one(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{first,rest}: Running{bytes,Machine{Sequence{[first]} <> rest,memo,capacity,next}}
    case stack: Halted{"TUPLE1 on an empty stack"}

def tuple_two(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{second,Con{first,rest}}: Running{bytes,Machine{Sequence{[first,second]} <> rest,memo,capacity,next}}
    case stack: Halted{"TUPLE2 on a short stack"}

def tuple_three(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{third,Con{second,Con{first,rest}}}: Running{bytes,Machine{Sequence{[first,second,third]} <> rest,memo,capacity,next}}
    case stack: Halted{"TUPLE3 on a short stack"}

def items_appended(parts: Split,bytes: List<&2,U32>,memo: Array<Value>,+capacity: U32,+next: U32) -> Step:
  match parts:
    case Unmarked{}: Halted{"APPENDS without a mark"}
    case Split{items,Con{Items{existing},rest}}: Running{bytes,Machine{Items{List.append(&2,Value,existing,items)} <> rest,memo,capacity,next}}
    case Split{items,rest}: Halted{"APPENDS on a value that is not a list"}

def appends(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  items_appended(split(stack,[]),bytes,memo,capacity,next)

def append(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{value,Con{Items{existing},rest}}: Running{bytes,Machine{Items{List.append(&2,Value,existing,[value])} <> rest,memo,capacity,next}}
    case stack: Halted{"APPEND on a value that is not a list"}

def entries_assigned(parts: Split,bytes: List<&2,U32>,memo: Array<Value>,+capacity: U32,+next: U32) -> Step:
  match parts:
    case Unmarked{}: Halted{"SETITEMS without a mark"}
    case Split{pairs,Con{Dict{entries},rest}}: Running{bytes,Machine{Dict{assign_pairs(pairs,entries)} <> rest,memo,capacity,next}}
    case Split{pairs,rest}: Halted{"SETITEMS on a value that is not a dictionary"}

def set_items(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,+capacity,+next} = machine
  entries_assigned(split(stack,[]),bytes,memo,capacity,next)

def set_item(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{value,Con{key,Con{Dict{entries},rest}}}: Running{bytes,Machine{Dict{assign(entries,key,value)} <> rest,memo,capacity,next}}
    case stack: Halted{"SETITEM on a value that is not a dictionary"}

# Objects and calls

def new_object(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{arguments,Con{class,rest}}: Running{bytes,Machine{Object{class,arguments,Absent{}} <> rest,memo,capacity,next}}
    case stack: Halted{"NEWOBJ on a short stack"}

def build(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{state,Con{Object{class,arguments,old},rest}}: Running{bytes,Machine{Object{class,arguments,state} <> rest,memo,capacity,next}}
    case stack: Halted{"BUILD on a value that is not an object"}

# An OrderedDict call with no arguments is a dictionary, so its items can follow.
def called(ordered: Bool,callable: Value,arguments: Value) -> Value:
  match ordered:
    case True{}: Dict{[]}
    case False{}: Reduced{callable,arguments}

def reduced(callable: Value,arguments: Value) -> Value:
  match callable arguments:
    case Global{+module,+name} Sequence{Nil{}}:
      called(String.eq(module,"collections") && String.eq(name,"OrderedDict"),Global{module,name},Sequence{[]})
    case callable arguments: Reduced{callable,arguments}

def reduce(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{arguments,Con{callable,rest}}: Running{bytes,Machine{reduced(callable,arguments) <> rest,memo,capacity,next}}
    case stack: Halted{"REDUCE on a short stack"}

def persistent(bytes: List<&2,U32>,machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{id,rest}: Running{bytes,Machine{Persistent{id} <> rest,memo,capacity,next}}
    case stack: Halted{"BINPERSID on an empty stack"}

def stop(machine: Machine) -> Step:
  Machine{stack,memo,capacity,next} = machine
  match stack:
    case Con{value,rest}: Completed{value}
    case stack: Halted{"STOP on an empty stack"}

def skip(count: Nat,bytes: List<&2,U32>,machine: Machine) -> Step:
  Running{List.drop(&2,U32,bytes,count),machine}

# One opcode and its argument bytes.
def operate(+code: U32,bytes: List<&2,U32>,machine: Machine) -> Step:
  match code:
    case 128: skip(1n,bytes,machine)
    case 149: skip(8n,bytes,machine)
    case 46: stop(machine)
    case 40: pushed(bytes,Mark{},machine)
    case 78: pushed(bytes,Absent{},machine)
    case 136: pushed(bytes,Flag{True{}},machine)
    case 137: pushed(bytes,Flag{False{}},machine)
    case 41: pushed(bytes,Sequence{[]},machine)
    case 93: pushed(bytes,Items{[]},machine)
    case 125: pushed(bytes,Dict{[]},machine)
    case 113: put_short(bytes,machine)
    case 114: put_long(bytes,machine)
    case 104: get_short(bytes,machine)
    case 106: get_long(bytes,machine)
    case 148: memoize(bytes,machine)
    case 88: text_long(bytes,machine)
    case 140: text_short(bytes,machine)
    case 84: text_long(bytes,machine)
    case 85: text_short(bytes,machine)
    case 99: global(bytes,machine)
    case 147: stack_global(bytes,machine)
    case 75: integer_short(bytes,machine)
    case 77: integer_two(bytes,machine)
    case 74: integer_four(bytes,machine)
    case 138: long_one(bytes,machine)
    case 71: float(bytes,machine)
    case 116: tuple(bytes,machine)
    case 133: tuple_one(bytes,machine)
    case 134: tuple_two(bytes,machine)
    case 135: tuple_three(bytes,machine)
    case 101: appends(bytes,machine)
    case 97: append(bytes,machine)
    case 117: set_items(bytes,machine)
    case 115: set_item(bytes,machine)
    case 129: new_object(bytes,machine)
    case 98: build(bytes,machine)
    case 82: reduce(bytes,machine)
    case 81: persistent(bytes,machine)
    case _: Halted{"opcode " ++ U32.show(code) ++ " is not supported"}

def step(bytes: List<&2,U32>,machine: Machine) -> Step:
  match bytes:
    case Nil{}: Halted{"ends before STOP"}
    case Con{code,rest}: operate(code,rest,machine)

# Every opcode takes at least one byte, so the byte count bounds the steps.
def run(fuel: Nat,state: Step) -> Parsed:
  match fuel state:
    case fuel Completed{value}: Parsed{value}
    case fuel Halted{detail}: Rejected{detail}
    case 0n Running{bytes,machine}: Rejected{"does not end"}
    case 1n+rest Running{bytes,machine}: run(rest,step(bytes,machine))

# A memo index is below the count of PUT opcodes, each of two bytes or more.
def parse(+bytes: List<&2,U32>) -> Parsed:
  +count = List.length(&2,U32,bytes)
  +depth = Buffers.depth_for((U32.from_nat(count) / 2 + 1 : U32))
  run(count,Running{bytes,Machine{[],Array.new(Value,depth,Absent{}),U32.shln(1,depth),0}})
