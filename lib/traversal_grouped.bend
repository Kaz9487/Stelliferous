# Shape-only groups, ordered folds from +0, then adjacent-pair trees. The
# reader is affine; the same traversal serves weight and bias gradients.
import Base
import ./traversal_loop.bend as Loop

type Range is Data:
  Range{first: Nat,count: Nat}

def rows(remaining: Nat,+first: Nat,+height: Nat,+width: Nat,+span: Nat) -> List<&2,Range>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Range{Nat.mul(first,width),Nat.mul(Nat.min(span,Nat.sub(height,first)),width)} <> rows(rest,Nat.add(first,span),height,width,span)

def columns(remaining: Nat,+row: Nat,+first: Nat,+width: Nat,+group: Nat) -> List<&2,Range>:
  match remaining:
    case 0n: []
    case 1n+rest:
      Range{Nat.add(Nat.mul(row,width),first),Nat.min(group,Nat.sub(width,first))} <> columns(rest,row,Nat.add(first,group),width,group)

def cat(left: List<&2,Range>,+right: List<&2,Range>) -> List<&2,Range>:
  match left:
    case []: right
    case head <> tail: head <> cat(tail,right)

def wide_rows(remaining: Nat,+row: Nat,+width: Nat,+group: Nat) -> List<&2,Range>:
  match remaining:
    case 0n: []
    case 1n+rest:
      cat(columns(Nat.div(Nat.add(width,Nat.sub(group,1n)),group),row,0n,width,group),wide_rows(rest,1n+row,width,group))

def ranges_selected(wide: Bool,+height: Nat,+width: Nat,+group: Nat) -> List<&2,Range>:
  match wide:
    case True{}: wide_rows(height,0n,width,group)
    case False{}:
      +span = Nat.div(group,width)
      rows(Nat.div(Nat.add(height,Nat.sub(span,1n)),span),0n,height,width,span)

def ranges(+height: Nat,+width: Nat,+group: Nat) -> List<&2,Range>:
  ranges_selected(Nat.is_lt(group,width),height,width,group)

def pairs(~ownership: Quant,~Element: Kind(ownership),~Context: Data,
  ~combine: Context -> Element -> Element -> Element,+context: Context,values: List<ownership,Element>) -> List<ownership,Element>:
  match values:
    case []: []
    case a <> []: [a]
    case a <> b <> tail: combine(context,a,b) <> pairs(~ownership,~Element,~Context,~combine,context,tail)

# The original length bounds the number of tree levels. A singleton is
# returned directly, including its sign bit; there is no final addition.
def levels(~ownership: Quant,~Element: Kind(ownership),~Context: Data,
  ~zero: Context -> Element,~combine: Context -> Element -> Element -> Element,
  fuel: Nat,+context: Context,values: List<ownership,Element>) -> Element:
  match fuel values:
    case 0n Nil{}: zero(context)
    case 0n head <> tail: head
    case 1n+rest Nil{}: zero(context)
    case 1n+rest head <> []: head
    case 1n+rest a <> b <> tail: levels(~ownership,~Element,~Context,~zero,~combine,rest,context,pairs(~ownership,~Element,~Context,~combine,context,a <> b <> tail))

def tree(+values: List<&2,F32>) -> F32:
  levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),List.length(&2,F32,values),Unit{},values)

type State<-Reader: Type,-Context: Data> is Type:
  State{reader: Reader,context: Context,position: Nat,total: F32}

def advanced(~Reader: Type,~Context: Data,+context: Context,position: Nat,total: F32,pair: Reader & F32) -> State<Reader,Context>:
  (reader,value) = pair
  State{reader,context,1n+position,(total + value : F32)}

def step(~Reader: Type,~Context: Data,~read: Reader -> Context -> Nat -> Reader & F32,state: State<Reader,Context>) -> State<Reader,Context>:
  State{reader,+context,+position,total} = state
  advanced(~Reader,~Context,context,position,total,read(reader,context,position))

def finished(~Reader: Type,~Context: Data,state: State<Reader,Context>) -> Reader & F32:
  State{reader,context,position,total} = state
  (reader,total)

def fold(~Reader: Type,~Context: Data,~read: Reader -> Context -> Nat -> Reader & F32,count: Nat,first: Nat,reader: Reader,context: Context) -> Reader & F32:
  finished(~Reader,~Context,Loop.run(~State<Reader,Context>,~step(~Reader,~Context,~read),count,State{reader,context,first,0.0}))

def prepend(~Reader: Type,value: F32,pair: Reader & List<&2,F32>) -> Reader & List<&2,F32>:
  (reader,values) = pair
  (reader,value <> values)

def grouped_tail(~Reader: Type,pair: Reader & F32,tail: Reader -> Reader & List<&2,F32>) -> Reader & List<&2,F32>:
  (reader,value) = pair
  prepend(~Reader,value,tail(reader))

def fold_ranges(~Reader: Type,~Context: Data,~read: Reader -> Context -> Nat -> Reader & F32,ranges: List<&2,Range>,reader: Reader,+context: Context) -> Reader & List<&2,F32>:
  match ranges:
    case []: (reader,[])
    case Range{first,count} <> rest:
      grouped_tail(~Reader,fold(~Reader,~Context,~read,count,first,reader,context),reader => fold_ranges(~Reader,~Context,~read,rest,reader,context))

def reduced(~Reader: Type,pair: Reader & List<&2,F32>) -> Reader & F32:
  (reader,values) = pair
  (reader,tree(values))

def run(~Reader: Type,~Context: Data,~read: Reader -> Context -> Nat -> Reader & F32,ranges: List<&2,Range>,reader: Reader,context: Context) -> Reader & F32:
  reduced(~Reader,fold_ranges(~Reader,~Context,~read,ranges,reader,context))
