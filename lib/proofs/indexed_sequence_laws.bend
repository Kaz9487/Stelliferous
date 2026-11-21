import Base
import ./natural_addition.bend as Addition
import ../math_sequence.bend as Sequence

def get_tabulated(~Element: Data,~Context: Data,~sample: Context -> Nat -> Element,+length: Nat,+index: Sequence.Index(length),+offset: Nat,
  +context: Context) -> {Sequence.get(Element,length,index,Sequence.tabulate(~Element,~Context,~sample,length,offset,context)) == sample(context,
    Nat.add(Sequence.index_value(length,index),offset)) : Element}:
  match length:
    case 0n:
      match index:
    case 1n+ +rest:
      match index:
        case Sequence.First{}: {==}
        case Sequence.Next{+previous}:
          %Addition.successor_right(Sequence.index_value(rest,previous),offset) : {Sequence.get(Element,rest,previous,Sequence.tabulate(~Element,~Context,
                ~sample,rest,1n+offset,context)) == sample(context,_) : Element}
          get_tabulated(~Element,~Context,~sample,rest,previous,1n+offset,context)
