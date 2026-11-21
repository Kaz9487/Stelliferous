# Shape-only groups and adjacent-pair trees, shared by the weight and bias
# gradients.
import Base

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
