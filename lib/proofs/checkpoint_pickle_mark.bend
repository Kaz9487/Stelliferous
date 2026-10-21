# Splitting the pickle stack at its mark loses nothing: the items it collects,
# put back in front of the mark and the rest, are the stack again.
import Base
import ../checkpoint_pickle.bend as Pickle

# The stack a split came from, or the fallback when there was no mark.
def restored(parts: Pickle.Split,fallback: List<&2,Pickle.Value>) -> List<&2,Pickle.Value>:
  match parts:
    case Pickle.Unmarked{}: fallback
    case Pickle.Split{items,rest}: List.reverse.go(&2,Pickle.Value,items,Pickle.Mark{} <> rest)

# The split of a stack below items already collected restores the whole stack.
def split_restores(stack: List<&2,Pickle.Value>,+items: List<&2,Pickle.Value>) ->
  {restored(Pickle.split(stack,items),List.reverse.go(&2,Pickle.Value,items,stack)) == List.reverse.go(&2,Pickle.Value,items,stack) : List<&2,Pickle.Value>}:
  match stack:
    case Nil{}: {==}
    case Con{Pickle.Mark{},rest}: {==}
    case Con{Pickle.Absent{},rest}: split_restores(rest,Pickle.Absent{} <> items)
    case Con{Pickle.Flag{value},rest}: split_restores(rest,Pickle.Flag{value} <> items)
    case Con{Pickle.Integer{word},rest}: split_restores(rest,Pickle.Integer{word} <> items)
    case Con{Pickle.Float{high,low},rest}: split_restores(rest,Pickle.Float{high,low} <> items)
    case Con{Pickle.Text{text},rest}: split_restores(rest,Pickle.Text{text} <> items)
    case Con{Pickle.Sequence{inner},rest}: split_restores(rest,Pickle.Sequence{inner} <> items)
    case Con{Pickle.Items{inner},rest}: split_restores(rest,Pickle.Items{inner} <> items)
    case Con{Pickle.Dict{entries},rest}: split_restores(rest,Pickle.Dict{entries} <> items)
    case Con{Pickle.Global{module,name},rest}: split_restores(rest,Pickle.Global{module,name} <> items)
    case Con{Pickle.Object{class,arguments,state},rest}: split_restores(rest,Pickle.Object{class,arguments,state} <> items)
    case Con{Pickle.Reduced{callable,arguments},rest}: split_restores(rest,Pickle.Reduced{callable,arguments} <> items)
    case Con{Pickle.Persistent{id},rest}: split_restores(rest,Pickle.Persistent{id} <> items)
